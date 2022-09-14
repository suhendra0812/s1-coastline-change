# import builtin packages
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
import xarray as xr

# import third party packages
from dateutil.parser import parse
from shapely.geometry import MultiLineString

# import created functions
from coastline_change_functions import (
    create_transects,
    segmentation,
    smooth_linestring,
    subpixel_contours,
    transect_analysis,
)

# setup logging
logger = logging.getLogger("s1_coastline_change")


def main() -> None:
    # define region id
    REGION_IDS = [678]
    # define list of tide type
    TIDE_TYPES = ["mean"]
    # define threhsold value
    THRESHOLD_VALUE = None  # None jika akan menggunakan threhsold otomatis, jika manual masukan angka nilai piksel

    # set working directory
    WORK_DIR = Path(__file__).parent.resolve()
    # define output directory
    OUTPUT_DIR = WORK_DIR / "output"
    # define s1 base directory
    DATASET_DIR = Path("/home/barata-serv/otomatisasi_barata/datasets")
    # define output dataset base directory
    OUTPUT_DATASET_DIR = DATASET_DIR / "s1" / "s1_gamma0_rtc" / "coastal"

    # define region path
    REGION_PATH = WORK_DIR / "region" / "coastal_grids.geojson"

    # read region file as GeoDataFrame
    region_gdf = gpd.read_file(REGION_PATH)

    # loop list of region ids
    for i, region_id in enumerate(REGION_IDS):
        # filter region
        selected_region = region_gdf.iloc[[region_id - 1]]

        # get geometry and wkt from geopandas object
        selected_region_geom = selected_region.geometry.unary_union
        selected_region_wkt = selected_region_geom.wkt

        # define suboutput directory
        suboutput_dir = OUTPUT_DIR / f"{region_id:04d}"
        suboutput_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Region ({i+1}/{len(selected_region)}) | ID: {region_id}")

        # loop tide type list
        for tide_type in TIDE_TYPES:
            logger.info(f"Tide type: {tide_type}")

            # read result path which has created from s1 downloader script
            result_path = suboutput_dir / f"{region_id:04d}_s1_{tide_type}.geojson"
            results = gpd.read_file(result_path)
            results.sort_values(by="startTimeFromAscendingNode", ignore_index=True)

            logger.info(f"Total s1 data: {len(results)}")

            da_list = []  # list dataset kosong

            # loop results
            for i, result in results.iterrows():
                # get information based on individual result
                platform_id = result["platform"]
                product_type = result["productType"]
                sensor_mode = result["sensorMode"]
                sensing_time = parse(result["startTimeFromAscendingNode"])

                year = sensing_time.year
                month = sensing_time.month
                day = sensing_time.day
                sensing_time_str = sensing_time.strftime("%Y%m%dT%H%M%S")

                # define dataset name
                dataset_name = f"{region_id:04d}_{sensing_time_str}_{platform_id}_{sensor_mode}_{product_type}_gamma0-rtc_VH.tif"

                # define dataset path
                dataset_path = OUTPUT_DATASET_DIR.joinpath(
                    f"{region_id:04d}", dataset_name
                )

                # load dataset using rioxarray
                da = rxr.open_rasterio(
                    dataset_path
                ).squeeze()  # membaca dataset geotif sebagai objek xarray, squeeze untuk menonaktifkan dimensi band
                da["band"] = "vh"  # mengganti nama band menjadi vh
                sensing_time_numpy = pd.to_datetime(
                    sensing_time
                ).to_numpy()  # konversi format waktu prerekaman jadi format numpy
                da = da.assign_coords(
                    time=sensing_time_numpy
                )  # menambahkan koordinat waktu ke xarray dataset
                da_list.append(da)  # dikumpulkan ke list dataset kosong di atas

            # list dataset di atas digabungkan dengan fungsi concat
            # agar jadi satu objek xarray dataset dengan dimensi waktu yang berbeda-beda
            vh_data = xr.concat(da_list, dim="time")

            logger.info("Segmentation...")
            vh_binary = vh_data.groupby("time").apply(
                lambda img: xr.apply_ufunc(  # apply_ufunc yaitu untuk menaplikasikan fungsi pada xarray
                    segmentation,  # fungsi yang diaplikasikan
                    kwargs={  # keyword arguments yang dimasukkan, karena fungsi segmentation meminta parameter 'img' dan 'threshold', maka kwargs-nya keduannya
                        "img": img,
                        "threshold": THRESHOLD_VALUE,
                    },
                )
            )

            logger.info("Extract coastline...")
            # fungsi subpixel_contour itu untuk mengekstrak garis kontur dari citra yang telah menjadi hitam putih (hasil segmentasi)
            # hasilnya disimpan dalam objek GeoDataFrame
            coastline_gdf = subpixel_contours(
                vh_binary,  # xarray hasil segmentasi yang hitam putih
                min_vertices=100,  # minimal jumlah titik vertek/node di dalam garis kontur yang nanti akan dihasilkan. Ini untuk meminimalisir deteksi kapal yang memiliki keliling yang kecil, tapi kelemahannya pulau kecil juga ikut hilang
                crs=vh_data.rio.crs,  # menentukan sistem projeksi yang digunakan. CRS (Coordinate Reference System)
                affine=vh_data.rio.transform(),  # menentukan transform (simplenya bounding box dari citra)
            )

            # dibawah ini adalah proses smoothing garis hasil ekstraksi proses di atas
            new_lines = []  # list garis baru kosong
            for (
                i,
                row,
            ) in coastline_gdf.iterrows():  # looping row dari GeoDataFrame garis
                line = (
                    row.geometry
                )  # ambil informasi geometry-nya dan disimpan dalam variable line
                if line.geom_type == "MultiLineString":
                    # jika jenis geometry MultiLineString / lebih dari satu garis
                    # dilakukan looping dulu terhadap multilinenya
                    # smoothing dilakukan di masing-masing garisnya
                    # parameter angka 5, itu simplenya tingkat kehalusan yang diiginkan
                    new_line = MultiLineString(
                        [smooth_linestring(l, 5) for l in line.geoms]
                    )
                else:
                    # jika satu garis, langsung dismoothing saja
                    new_line = smooth_linestring(line, 5)
                new_lines.append(
                    new_line
                )  # tambahkan garis yang telah dihaluskan ke list garis baru

            coastline_gdf.geometry = (
                new_lines  # ganti garis yang ada di geometry menjadi garis baru
            )

            # ini proses pembuatan transek
            # menentukan baseline menggunakan garis pantai terlama
            baseline = coastline_gdf.geometry.iloc[0]
            # membuat transek berdasarkan baseline,
            # dengan jarak antar transek 500 m
            # panjang transek 100 ke darat dan 100 ke laut jadi total 200 m
            # tentukan sistem proyeksinya
            # hasilnya disimpan ke dalam objek GeoDataFrame
            transect_gdf = create_transects(baseline, 500, 100, crs=coastline_gdf.crs)

            logger.info("Transect analysis...")
            # ini proses analisis perubahan garis pantainya metode transek
            # parameternya:
            # 1. coastline_gdf: GeoDataFrame garis pantai
            # 2. transect_gdf: GeoDataFrame transek
            # 3. "time": kolom waktu pada coastline_gdf yang akan digunakan untuk perhitungan waktu antar garis pantai
            # 4. reverse=True: kadang ada kesalahan arah perubahan garis pantai, sehingga perlu di-reverse. Hal ini perlu dikroscek di QGIS dengan symbologi arrow, ketika hasilnya keluar
            # hasilnya disimpan dalam objek GeoDataFrame
            transect_analysis_gdf = transect_analysis(
                coastline_gdf, transect_gdf, "time", reverse=True
            )

            # di bawabh ini untuk menyimpan semua hasilnya ke dalam file .geojson
            coastline_path = (
                suboutput_dir / f"{region_id:04d}_s1_{tide_type}_coastlines.geojson"
            )
            transect_path = (
                suboutput_dir / f"{region_id:04d}_s1_{tide_type}_transects.geojson"
            )
            transect_analysis_path = (
                suboutput_dir
                / f"{region_id:04d}_s1_{tide_type}_transect_analysis.geojson"
            )

            coastline_gdf.to_file(coastline_path, driver="GeoJSON")
            transect_gdf.to_file(transect_path, driver="GeoJSON")
            transect_analysis_gdf.to_file(transect_analysis_path, driver="GeoJSON")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    main()
