import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import stackstac
import xarray as xr
from dask.distributed import Client
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from pystac import ItemCollection
from rioxarray.merge import merge_arrays
from shapely.geometry import MultiLineString, mapping

from big_tide_prediction import tide_interpolation, tide_prediction
from coastline_change_functions import (
    coregistration,
    create_transects,
    db_scale,
    filter_tide,
    intersection_percent,
    lee_filter,
    rescale,
    segmentation,
    smooth_linestring,
    subpixel_contours,
    transect_analysis,
)

logger = logging.getLogger("s1_coastline_change_stac")


def main() -> None:
    REGION_IDS = [713]
    PROVINCE = "JAWA TIMUR" # if defined, REGION_IDS will be overwritten. Set to None to use defined REGION_IDS.
    TIDE_TYPES = ["mean"]
    CLIENT_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    COLLECTION = "sentinel-1-rtc"
    START_DATE = "2015-01-01"
    STOP_DATE = "2022-12-31"
    MIN_AREA_PERCENT = 98  # percent
    COREGISTRATION = False
    THRESHOLD = None

    region_path = Path("./region/coastal_grids.geojson")
    point_path = Path("./region/coastal_points.geojson")

    with Client(n_workers=4, threads_per_worker=2, memory_limit="4GB") as dask_client:
        logger.info(f"Dask client dashboard link: {dask_client.dashboard_link}")

        region_gdf = gpd.read_file(region_path)
        point_gdf = gpd.read_file(point_path)

        if PROVINCE:
            REGION_IDS = [i+1 for i in region_gdf.query("province == @PROVINCE").index.tolist()]

        for i, region_id in enumerate(REGION_IDS):
            logger.info(f"({i+1}/{len(REGION_IDS)}) Region ID: {region_id}")
            selected_region_gdf = region_gdf.loc[[region_id - 1]]
            selected_point_gdf = point_gdf.loc[[region_id - 1]]

            output_dir = Path("./output") / f"{region_id:04d}"
            output_dir.mkdir(parents=True, exist_ok=True)

            xmin, ymin, xmax, ymax = selected_region_gdf.total_bounds.tolist()
            bbox = xmin, ymin, xmax, ymax

            catalog = pystac_client.Client.open(CLIENT_URL)

            area_list = []
            item_list = []

            start_date = parse(START_DATE)
            while start_date <= parse(STOP_DATE):
                stop_date = start_date + relativedelta(years=1)
                logger.info(f"Datetime search: {start_date} - {stop_date}")

                query = catalog.search(
                    collections=[COLLECTION],
                    datetime=[start_date, stop_date],
                    bbox=bbox,
                    query={"sar:polarizations": {"eq": ["VV", "VH"]}},
                )
                items = query.get_items()
                for item in items:
                    area = intersection_percent(
                        item, mapping(selected_region_gdf.unary_union)
                    )
                    if area >= MIN_AREA_PERCENT:
                        area_list.append(area)
                        item_list.append(item)

                start_date = stop_date

            item_list = sorted(item_list, key=lambda x: x.datetime)
            s1_items = ItemCollection(item_list)
            logger.info(f"S1 Found: {len(s1_items)} datasets")

            signed_s1_items = [pc.sign(item).to_dict() for item in s1_items]

            s1_data = (
                stackstac.stack(
                    signed_s1_items,
                    bounds_latlon=bbox,
                    epsg=3857,
                    resolution=10,
                )
                # .where(lambda x: x > 0, other=np.nan)
                .sel(band="vh").rio.write_nodata(0)
            )

            dem_query = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)

            dem_items = dem_query.get_all_items()
            logger.info(f"DEM found: {len(dem_items):d} datasets")

            signed_dem_items = [pc.sign(item).to_dict() for item in dem_items]

            dem_data = (
                stackstac.stack(signed_dem_items, bounds_latlon=bbox, epsg=3857)
                # .where(lambda x: x > 0, other=np.nan)
                .sel(band="data").rio.write_nodata(0)
            )

            merged_dem_data = merge_arrays([dem for dem in dem_data.load()])
            merged_dem_data

            times = s1_data.time.values
            logger.info(f"Time count: {len(times)}")

            x = selected_point_gdf.unary_union.centroid.x
            y = selected_point_gdf.unary_union.centroid.y

            tide_path = output_dir / f"{region_id:04d}_tide.csv"

            if not tide_path.exists():
                start_date = pd.to_datetime(times)[0].date()
                stop_date = pd.to_datetime(times)[-1].date()
                tide_df = tide_prediction(x, y, start_date, stop_date)
                interp_tide_df = tide_interpolation(
                    tide_df, pd.to_datetime(times).tolist()
                )
                interp_tide_df.to_csv(tide_path, index=False)
            else:
                interp_tide_df = pd.read_csv(tide_path)

            tide_list = interp_tide_df["level"].tolist()
            logger.info(f"Tide count: {len(tide_list)}")

            lt = np.min(tide_list)
            ht = np.max(tide_list)
            mean = np.mean(tide_list)

            logger.info(f"Low tide: {lt}")
            logger.info(f"High tide: {ht}")
            logger.info(f"Mean tide: {mean}")

            tide_data = xr.DataArray(tide_list, coords=[s1_data.time], dims=["time"])

            s1_data["tide"] = tide_data

            group_s1_data = s1_data.groupby("time.year")

            tide_dict = {
                "lt": lt,
                "mean": mean,
                "ht": ht
            }

            for tide_type in TIDE_TYPES:
                logger.info(f"Filter data by tide type: {tide_type}...")

                tide_s1_data = filter_tide(group_s1_data, tide_dict[tide_type])
                logger.info(f"Filtered tide S1 data count: {tide_s1_data.shape[0]}")

                suboutput_dir = output_dir / tide_type
                suboutput_dir.mkdir(parents=True, exist_ok=True)

                output_tifs = sorted(suboutput_dir.glob("*.tif"))
                if len(output_tifs) == len(tide_s1_data.time):
                    continue

                logger.info("Load data from dask client...")
                vh_data = tide_s1_data.load()

                if COREGISTRATION:
                    logger.info(f"Coregistration...")
                    new_vh_data = (
                        coregistration(vh_data)
                        .groupby("time")
                        .apply(lambda x: x.rio.write_nodata(0).rio.interpolate_na())
                    )
                else:
                    new_vh_data = vh_data.copy()

                logger.info("Convert to dB...")
                vh_db_data = new_vh_data.groupby("time").apply(db_scale)

                logger.info("Speckle filter...")
                vh_filter = vh_db_data.groupby("time").apply(
                    lambda img: xr.apply_ufunc(
                        lee_filter,
                        img,
                        kwargs={"size": 5},
                        # dask="parallelized",
                        # dask_gufunc_kwargs={"allow_rechunk": True}
                    )
                )

                threshold_list = [THRESHOLD for _ in range(len(vh_filter.time))]

                vh_filter["threshold"] = xr.DataArray(
                    threshold_list, coords=[vh_filter.time], dims=["time"]
                )

                logger.info("Segmentation...")
                vh_binary = vh_filter.groupby("time").apply(
                    lambda img: xr.apply_ufunc(
                        segmentation,
                        kwargs={"img": img, "threshold": img.threshold.values}
                        # img.chunk({"x": -1, "y": -1}),
                        # dask="parallelized",
                    )
                )

                dem_regrid = merged_dem_data.interp_like(vh_binary.isel(time=-1))
                dem_regrid = dem_regrid > 30

                vh_binary_filtered = vh_binary.groupby("time").apply(
                    lambda x: x.where(~dem_regrid, other=1)
                )

                vh_binary_filtered.isel(time=1).plot(cmap="gray", size=10)

                logger.info("Extract coastline...")
                coastline_gdf = subpixel_contours(
                    vh_binary_filtered,
                    min_vertices=100,
                    crs=s1_data.crs,
                    affine=s1_data.transform,
                )

                new_lines = []
                for i, row in coastline_gdf.iterrows():
                    line = row.geometry
                    if line.geom_type == "MultiLineString":
                        new_line = MultiLineString(
                            [smooth_linestring(l, 5) for l in line.geoms]
                        )
                    else:
                        new_line = smooth_linestring(line, 5)
                    new_lines.append(new_line)

                coastline_gdf.geometry = new_lines

                baseline = coastline_gdf.geometry.iloc[0]
                transect_gdf = create_transects(
                    baseline, 500, 100, crs=coastline_gdf.crs
                )

                logger.info("Transect analysis...")
                transect_analysis_gdf = transect_analysis(
                    coastline_gdf, transect_gdf, "time", reverse=True
                )

                if transect_analysis_gdf is None:
                    continue
                
                coastline_path = (
                    suboutput_dir / f"{region_id:04d}_s1_coastlines.geojson"
                )
                transect_path = suboutput_dir / f"{region_id:04d}_s1_transects.geojson"
                transect_analysis_path = (
                    suboutput_dir / f"{region_id:04d}_s1_transect_analysis.geojson"
                )

                coastline_gdf.to_file(coastline_path, driver="GeoJSON")
                transect_gdf.to_file(transect_path, driver="GeoJSON")
                transect_analysis_gdf.to_file(transect_analysis_path, driver="GeoJSON")

                vh_db_rescale = vh_db_data.groupby("time").apply(
                    rescale,
                    target_type_min=1,
                    target_type_max=255,
                    target_type=np.uint8,
                )

                if COREGISTRATION:
                    suffix = "_coreg.tif"
                else:
                    suffix = ".tif"

                for time, d in vh_db_rescale.rename("vh_db").groupby("time"):
                    year = pd.to_datetime(time).year
                    raster_path = (
                        suboutput_dir / f"{region_id:04d}_{year}_s1_vh_db{suffix}"
                    )
                    d.rio.to_raster(raster_path, crs=s1_data.crs, compress="lzw")
                    logger.info(f"Saved to {raster_path}")

                del vh_data
                del new_vh_data
                del vh_db_data
                del vh_filter
                del vh_binary
                del vh_binary_filtered
                del dem_regrid
                del vh_db_rescale

            del merged_dem_data
            del tide_data
            del s1_data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    main()
