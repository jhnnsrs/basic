from datetime import datetime
import logging
import math
import os
import uuid
from enum import Enum
from typing import List, Optional

import numpy as np
import tifffile
import xarray as xr
from colorthief import ColorThief
from matplotlib import cm
from PIL import Image
from rich.logging import RichHandler
from xarray_multiscale import multiscale
from xarray_multiscale.reducers import windowed_mean

from mikro.api.schema import (
    ExperimentFragment,
    MetricFragment,
    OmeroFileFragment,
    OmeroFileType,
    RepresentationFragment,
    RepresentationVariety,
    ROIFragment,
    ROIType,
    SampleFragment,
    Search_representationQuery,
    Search_sampleQuery,
    ThumbnailFragment,
    OmeroRepresentationInput,
    RoiTypeInput,
    aexpand_omerofile,
    aexpand_representation,
    aexpand_sample,
    aexpand_thumbnail,
    create_experiment,
    create_metric,
    create_sample,
    create_roi,
    create_thumbnail,
    from_xarray,
    get_representation,
    InputVector,
)
from arkitekt import register

logging.basicConfig(level="INFO", handlers=[RichHandler()])

logger = logging.getLogger(__name__)


class Colormap(Enum):
    VIRIDIS = cm.viridis
    PLASMA = cm.plasma


@register()
def array_to_image(
    rep: RepresentationFragment,
    rescale=True,
    max=True,
    cm: Colormap = Colormap.VIRIDIS,
) -> ThumbnailFragment:
    """Thumbnail Image

    Generates THumbnail for the Image

    Args:
        rep (Representation): The to be converted Image
        rescale (bool, optional): SHould we rescale the image to fit its dynamic range?. Defaults to True.
        max (bool, optional): Automatically z-project if stack. Defaults to True.
        cm: (Colormap, optional): The colormap to use. Defaults to Colormap.VIRIDIS.

    Returns:
        Thumbnail: The Thumbnail
    """

    array = rep.data

    if "z" in array.dims:
        if not max:
            raise Exception("Set Max to Z true if you want to convert image stacks")
        array = array.max(dim="z")

    if "t" in array.dims:
        array = array.sel(t=0)

    if "c" in array.dims:
        # Check if we have to convert to monoimage
        if array.c.size == 1:
            array = array.sel(c=0)

            if rescale == True:
                logger.info("Rescaling")
                min, max = array.min(), array.max()
                image = np.interp(array, (min, max), (0, 255)).astype(np.uint8)
            else:
                image = (array * 255).astype(np.uint8)

            print(cm)
            mapped = cm(image)

            finalarray = (mapped * 255).astype(np.uint8)

        else:
            if array.c.size >= 3:
                array = array.sel(c=[0, 1, 2]).transpose(*list("xyc")).data
            elif array.c.size == 2:
                # Two Channel Image will be displayed with a Dark Channel
                array = np.concatenate(
                    [
                        array.sel(c=[0, 1]).transpose(*list("xyc")).data,
                        np.zeros((array.x.size, array.y.size, 1)),
                    ],
                    axis=2,
                )

            if rescale == True:
                logger.info("Rescaling")
                min, max = array.min(), array.max()
                finalarray = np.interp(array.compute(), (min, max), (0, 255)).astype(
                    np.uint8
                )
            else:
                finalarray = (array.compute() * 255).astype(np.uint8)

    else:
        raise NotImplementedError("Image Does not provide the channel Argument")

    temp_file = uuid.uuid4().hex + ".png"

    img = Image.fromarray(finalarray)
    img = img.convert("RGB")
    img.save(temp_file, "PNG")

    color_thief = ColorThief(temp_file)

    th = create_thumbnail(
        file=open(temp_file, "rb"),
        rep=rep,
        major_color="#%02x%02x%02x" % color_thief.get_color(quality=3),
    )
    os.remove(temp_file)
    return th


@register()
def convert_tiff_file(
    file: OmeroFileFragment,
    experiment: Optional[ExperimentFragment],
    sample: Optional[SampleFragment],
    auto_create_sample: bool = True,
) -> RepresentationFragment:
    """Convert File

    Converts an a Mikro File in a Usable zarr based Image

    Args:
        file (OmeroFileFragment): The File to be converted
        sample (Optional[SampleFragment], optional): The Sample to which the Image belongs. Defaults to None.
        experiment (Optional[ExperimentFragment], optional): The Experiment to which the Image belongs. Defaults to None.
        auto_create_sample (bool, optional): Should we automatically create a sample if none is provided?. Defaults to True.

    Returns:
        Representation: The Back
    """

    assert (
        file.type == OmeroFileType.TIFF
    ), "Cannot convert anything but tiffs at the moment"

    print(experiment)

    if not sample and auto_create_sample:
        sample = create_sample(
            name=file.name, experiments=[experiment] if experiment else []
        )

        print(sample)

    assert file.file, "No File Provided"
    with file.file as f:
        image = tifffile.imread(f)

        print(image.shape)

        image = image.reshape((1,) * (5 - image.ndim) + image.shape)
        array = xr.DataArray(image, dims=list("ctzyx"))

    omero = OmeroRepresentationInput(acquisitionDate=datetime.now())

    return from_xarray(
        array,
        name=file.name,
        sample=sample,
        file_origins=[file],
        omero=omero,
        tags=["converted"],
    )


@register()
def measure_max(
    rep: RepresentationFragment,
    key: str = "max",
) -> MetricFragment:
    """Measure Max

    Measures the maxium value of an image

    Args:
        rep (OmeroFiRepresentationFragmentle): The image
        key (str, optional): The key to use for the metric. Defaults to "max".

    Returns:
        Representation: The Back
    """
    return create_metric(
        key=key, value=float(rep.data.max().compute()), representation=rep
    )


@register()
def measure_basics(
    rep: RepresentationFragment,
) -> List[MetricFragment]:
    """Measure Basic Metrics

    Measures basic metrics of an image like max, min, mean

    Args:
        rep (OmeroFiRepresentationFragmentle): The image

    Returns:
        Representation: The Back
    """

    x = rep.data.compute()

    return [
        create_metric(key="maximum", value=float(x.max()), representation=rep),
        create_metric(key="mean", value=float(x.mean()), representation=rep),
        create_metric(key="min", value=float(x.min()), representation=rep),
    ]


@register()
def t_to_frame(
    rep: RepresentationFragment,
    interval: int = 1,
    key: str = "frame",
) -> ROIFragment:
    """T to Frame

    Converts a time series to a single frame

    Args:
        rep (RepresentationFragment): The Representation
        frame (int): The frame to select

    Returns:
        RepresentationFragment: The new Representation
    """
    assert "t" in rep.data.dims, "Cannot convert non time series to frame"

    for i in range(rep.data.sizes["t"]):
        if i % interval == 0:
            yield create_roi(
                representation=rep,
                label=f"{key} {i}",
                type=RoiTypeInput.FRAME,
                tags=[f"t{i}", "frame"],
                vectors=[InputVector(t=i), InputVector(t=i + interval)],
            )


@register()
def z_to_slice(
    rep: RepresentationFragment,
    interval: int = 1,
    key: str = "Slice",
) -> ROIFragment:
    """Z to Slice

    Creates a slice roi for each z slice

    Args:
        rep (RepresentationFragment): The Representation
        frame (int): The frame to select

    Returns:
        RepresentationFragment: The new Representation
    """
    assert "z" in rep.data.dims, "Cannot convert non time series to frame"

    for i in range(rep.data.sizes["z"]):
        if i % interval == 0:
            yield create_roi(
                representation=rep,
                label=f"{key} {i}",
                type=RoiTypeInput.SLICE,
                tags=[f"z{i}", "frame"],
                vectors=[InputVector(z=i), InputVector(z=i + interval)],
            )


@register()
def crop_image(
    roi: ROIFragment, rep: Optional[RepresentationFragment]
) -> RepresentationFragment:
    """Crop Image

    Crops an Image based on a ROI

    Args:
        roi (ROIFragment): The Omero File
        rep (Optional[RepresentationFragment], optional): The Representation to be cropped. Defaults to the one of the ROI.

    Returns:
        Representation: The Back
    """
    if rep == None:
        rep = get_representation(roi.representation.id)

    array = rep.data
    if roi.type == ROIType.RECTANGLE:
        x_start = roi.vectors[0].x
        y_start = roi.vectors[0].y
        x_end = roi.vectors[0].x
        y_end = roi.vectors[0].y

        for vector in roi.vectors:
            if vector.x < x_start:
                x_start = vector.x
            if vector.x > x_end:
                x_end = vector.x
            if vector.y < y_start:
                y_start = vector.y
            if vector.y > y_end:
                y_end = vector.y

        roi.vectors[0]

        array = array.sel(
            x=slice(math.floor(x_start), math.floor(x_end)),
            y=slice(math.floor(y_start), math.floor(y_end)),
        )

        return from_xarray(
            array,
            name="Cropped " + rep.name,
            tags=["cropped"],
            origins=[rep],
            roi_origins=[roi],
        )

    if roi.type == ROIType.FRAME:
        array = array.sel(
            t=slice(math.floor(roi.vectors[0].t), math.floor(roi.vectors[1].t))
        )

        return from_xarray(
            array,
            name="Cropped " + rep.name,
            tags=["cropped"],
            origins=[rep],
            roi_origins=[roi],
        )

    if roi.type == ROIType.SLICE:
        array = array.sel(
            z=slice(math.floor(roi.vectors[0].z), math.floor(roi.vectors[1].z))
        )

        return from_xarray(
            array,
            name="Cropped " + rep.name,
            tags=["cropped"],
            origins=[rep],
            roi_origins=[roi],
        )

    raise Exception(f"Roi Type {roi.type} not supported")


class DownScaleMethod(Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"


@register()
def downscale_image(
    rep: RepresentationFragment,
    factor: int = 2,
    depth=0,
    method: DownScaleMethod = DownScaleMethod.MEAN,
) -> RepresentationFragment:
    """Downscale

    Scales down the Representatoi by the factor of the provided

    Args:
        rep (RepresentationFragment): The Image where we should count cells

    Returns:
        RepresentationFragment: The Downscaled image
    """
    s = tuple([1 if c == 1 else factor for c in rep.data.squeeze().shape])

    newrep = multiscale(rep.data.squeeze(), windowed_mean, s)

    return from_xarray(
        newrep[1],
        name=f"Downscaled {rep.name} by {factor}",
        tags=[f"scale-{factor}"],
        variety=RepresentationVariety.VOXEL,
        origins=[rep],
    )
