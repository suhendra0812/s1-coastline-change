import logging
from pathlib import Path

import numpy as np
import pyTMD

logger = logging.getLogger("model_tide_prediction")


def get_constants(lon: np.array, lat: np.array, model: pyTMD.model) -> tuple:

    logger.info("Extracting constants...")

    # get amplitude and phase
    amp, ph = pyTMD.extract_FES_constants(
        np.atleast_1d(lon),
        np.atleast_1d(lat),
        model.model_file,
        TYPE=model.type,
        VERSION=model.version,
        METHOD="spline",
        EXTRAPOLATE=True,
        SCALE=model.scale,
        GZIP=model.compressed,
    )

    return amp, ph


def model_tide_prediction(
    lon: np.array, lat: np.array, date_list: np.array, model_dir: Path
) -> np.array:

    logger.info("Tide prediction")

    # convert list of datetime
    tide_time = pyTMD.time.convert_datetime(date_list)

    # define model directory and initialize model based on model format
    model = pyTMD.model(model_dir, format="FES", compressed=False).elevation("FES2014")

    # get tide constants (amplitude and phase) and it will take a while
    amp, ph = get_constants(lon, lat, model)

    # extract model constituent
    c = model.constituents

    # calculate delta time
    delta_file = pyTMD.utilities.get_data_path(["data", "merged_deltat.data"])
    DELTAT = pyTMD.calc_delta_time(delta_file, tide_time)

    # calculate complex phase in radians for Euler's
    cph = -1j * ph * np.pi / 180.0

    # calculate constituent oscillation
    hc = amp * np.exp(cph)

    # predict tidal time series
    TIDE = pyTMD.predict_tidal_ts(
        tide_time, hc, c, DELTAT=DELTAT, CORRECTIONS=model.format
    )

    # infer minor corrections
    MINOR = pyTMD.infer_minor_corrections(
        tide_time, hc, c, DELTAT=DELTAT, CORRECTIONS=model.format
    )

    # calculate tide with minor correction
    TIDE.data[:] += MINOR.data[:]

    logger.info("Done")

    return TIDE
