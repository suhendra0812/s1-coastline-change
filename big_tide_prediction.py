from datetime import date, datetime
from typing import List
import httpx
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("big_tide_prediction")

def tide_prediction(
    lon: float, lat: float, start_date: date, stop_date: date
) -> pd.DataFrame:
    logger.info("Tide predicting...")
    url = "https://srgi.big.go.id/tides_data/prediction-v2"

    params = {
        "coords": f"{lon},{lat}",
        "awal": f"{start_date.isoformat()}",
        "akhir": f"{stop_date.isoformat()}",
    }
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params=params)

        results = r.json()["results"]
        predictions = results["predictions"]
        ids = [i for i in predictions.keys()]
        values = [v for v in predictions.values()]
        error_list = ["Site", "is", "out", "of", "model", "grid", "OR", "land"]
        if set(error_list).issubset(values[0]):
            raise ValueError(" ".join(error_list))
        df = pd.DataFrame(data=values, index=pd.Index(ids))
        df.columns = ["lat", "lon", "date", "time", "level"]
        df["lat"] = df["lat"].astype(float)
        df["lon"] = df["lon"].astype(float)
        df["level"] = df["level"].astype(float)
        df["datetime"] = pd.to_datetime(
            df["date"].str.cat(df["time"], sep="T"), utc=True
        )

    return df


def tide_interpolation(
    tide_df: pd.DataFrame, datetime_list: List[datetime]
) -> pd.DataFrame:
    tide_df.set_index("datetime", inplace=True)
    index_list = pd.DatetimeIndex(
        tide_df.index.tolist() + pd.to_datetime(datetime_list, utc=True).tolist()
    )
    interp_df = tide_df.copy()
    interp_df = interp_df.reindex(index_list)
    interp_df = interp_df[["level"]].interpolate(method="time")
    interp_df = interp_df.loc[pd.to_datetime(datetime_list, utc=True)]
    interp_df.sort_index(inplace=True)
    interp_df["lat"] = np.repeat(tide_df["lat"].unique(), len(interp_df))
    interp_df["lon"] = np.repeat(tide_df["lon"].unique(), len(interp_df))
    interp_df.reset_index(inplace=True)
    interp_df.rename(columns={"index": "datetime"}, inplace=True)
    logger.info("Tide interpolated")
    return interp_df


def main() -> None:
    lon, lat = 113.7162,-7.5433
    date_range = pd.date_range("2015-01-01", "2022-08-12", freq="2M")
    start_date = date_range[0]
    stop_date = date_range[-1]

    tide_df = tide_prediction(lon, lat, start_date, stop_date)
    interp_tide_df = tide_interpolation(tide_df, date_range.tolist())
    interp_tide_df.to_csv("./tide.csv", index=False)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
