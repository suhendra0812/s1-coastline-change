# modul bawaan python
from datetime import datetime
import itertools
import logging
from pathlib import Path

# modul pihak ketiga
from dateutil.parser import parse
from eodag import EODataAccessGateway, SearchResult
import geopandas as gpd
import numpy as np
import pandas as pd

# modul buatan sendiri
from big_tide_prediction import tide_prediction, tide_interpolation


# menyiapkan logging
logger = logging.getLogger(__name__) # nama loggingnya yang akan dipanggil setiap kali akan dilakukan logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
) # logging sampai level INFO dengan format 'waktu [level] nama proses: pesan'

SEARCH_PRODUCT_TYPE = "S1_SAR_GRD" # produk tipe yang diinginkan yang bisa didapat dengan memanggil 'eodag list' di terminal
SEARCH_START_DATE = "2015-01-01" # tanggal awal dengan format 'yyyy-mm-dd'
SEARCH_END_DATE = datetime.today().date().isoformat() # sama dengan tanggal awal, tapi didapat dari cara yang berbeda yaitu otomatis dari method 'today()', 'isoformat()' yaitu untuk memformat tanggal menjadi format 'yyyy-mm-dd'
MIN_AREA = 98 # persentase area minimal yang digunakan untuk memfilter data sentinel-1 yang akan didownload
REGION_IDS = [671] # list region id yang akan dilakukan analisis. Untuk menambahkan region id lebih dari satu [0000, 0001, ...]. Nanti akan dilooping satu per satu
TIDE_TYPES = ["mean"] # list tipe pasut yang akan digunakan. Tipe pasut digunakan untuk filter data sentinel-1 yang memiliki kesamaan tipe pasut. Untuk menambahkan tipe pasut lebih dari satu ["mean", "ht", "lt"]. Nanti akan dilooping satu per satu.
OUTPUT_DIR = Path("./output").resolve() # folder untuk menyimpan metadata sentinel-1 dalam bentuk geojson
S1_BASE_DIR = Path("/home/barata-serv/otomatisasi_barata/datasets") # folder untuk menyimpan data sentinel-1

REGION_PATH = Path("./region/coastal_grids.geojson").resolve() # data polygon grid sebagai AOI pencarian data sentinel-1
POINT_PATH = Path("./region/coastal_points.geojson").resolve() # data point pada setiap grid untuk prediksi pasut
PROVIDER_PATH = Path("./provider/sara_provider.yaml").resolve() # metadata penyedia (provider) layanan download data citra satelit (termasuk sentinel-1) dari australia yang namanya SARA (Sentinel Australasia Regional Access)

dag = EODataAccessGateway() # membuat objek eodag baru untuk digunakan selanjutnya saat memanggil fungsi-fungsi seperti pencarian dan download data sentinel

with open(PROVIDER_PATH) as f: # membuka file provider
    dag.update_providers_config(f.read()) # karena provider SARA belum tersedia di eodag, maka perlu ditambahkan, tapi belum disetting sebagai provider default

assert "sara" in dag.available_providers() # hanya untuk mengecek provider 'sara' sudah tersedia atau tidak

dag.set_preferred_provider("sara") # setting 'sara' sebagai provider default

region_gdf = gpd.read_file(REGION_PATH) # membaca file region sebagai GeoDataFrame

point_gdf = gpd.read_file(POINT_PATH)  # membaca file point sebagai GeoDataFrame

for i, region_id in enumerate(REGION_IDS): # looping region id
    selected_region = region_gdf.iloc[[region_id - 1]] # filter region berdasarkan region id
    selected_point = point_gdf.iloc[[region_id - 1]] # filter point berdasarkan region id

    selected_region_geom = selected_region.geometry.unary_union # mengambil informasi geometry dari region

    suboutput_dir = OUTPUT_DIR / f"{region_id:04d}" # subfolder di dalam folder output dengan menambahkan region id
    suboutput_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Region ({i+1}/{len(selected_region)}) | ID: {region_id}")

    # mendefinisikan parameter pencarian
    search_params = {
        "productType": SEARCH_PRODUCT_TYPE,
        "start": SEARCH_START_DATE,
        "end": SEARCH_END_DATE,
        "geom": selected_region_geom,
    }

    # run search all based on search parameter
    search_results = (
        dag.search_all(**search_params)
        .filter_property(polarizationMode="VH,VV") # filter polarisasi yang ada VH-nya
        .filter_overlap(selected_region_geom, minimum_overlap=MIN_AREA) # filter area overlap scene dengan AOI berdasarkan persentase MIN_AREA
    )

    # get datetime list from search results
    time_list = sorted(
        [
            pd.to_datetime(result.properties["startTimeFromAscendingNode"], utc=True)
            for result in search_results
        ]
    )

    # get x and y from filtered point
    x = selected_point.unary_union.centroid.x
    y = selected_point.unary_union.centroid.y

    # define tide output path
    tide_path = suboutput_dir / f"{region_id:04d}_tide.csv"

    # tide prediction based on big website
    if not tide_path.exists():
        # get minimum and maximum of date in the data
        start_date_data = time_list[0]
        stop_date_data = time_list[-1]

        # get tide prediction based on x, y and date range (start and stop date)
        tide_df = tide_prediction(x, y, start_date_data, stop_date_data)

        # interpolate tide data based on sentinel-1 time list and save it into csv
        interp_tide_df = tide_interpolation(tide_df, time_list)
        interp_tide_df.to_csv(tide_path, index=False)
    else:
        # if tide csv file is existed, read it instead
        interp_tide_df = pd.read_csv(tide_path)

    # listing tide elevation data
    tide_list = interp_tide_df["level"].tolist()

    # calculate low, high and mean of tide list
    lt = np.min(tide_list)
    ht = np.max(tide_list)
    mean = np.mean(tide_list)

    logger.info(f"Low tide: {lt} | High tide: {ht} | Mean tide: {mean}")

    # pair between search results and tide list and put tide into search results
    tide_results = []
    for result, tide in zip(search_results, tide_list):
        result.properties["tide"] = tide
        tide_results.append(result)
    
    # convert list of results to SearchResult object
    tide_results = SearchResult(
        sorted(tide_results, key=lambda x: x.properties["startTimeFromAscendingNode"])
    )

    # grouping search results based on year
    grouped_results = [
        SearchResult(group)
        for _, group in itertools.groupby(
            tide_results,
            key=lambda x: parse(x.properties["startTimeFromAscendingNode"]).year,
        )
    ]

    # categorizing search results into 3 different results based on tide data
    lt_results = SearchResult(
        [
            sorted(group, key=lambda x: abs(x.properties["tide"] - lt))[0]
            for group in grouped_results
        ]
    )
    ht_results = SearchResult(
        [
            sorted(group, key=lambda x: abs(x.properties["tide"] - ht))[0]
            for group in grouped_results
        ]
    )
    mean_results = SearchResult(
        [
            sorted(group, key=lambda x: abs(x.properties["tide"] - mean))[0]
            for group in grouped_results
        ]
    )

    # mapping each result based on tide type keys
    tide_s1_results_dict = {"ht": ht_results, "lt": lt_results, "mean": mean_results}

    # filter results based on defined tide type
    tide_s1_results_dict = {
        key: value for key, value in tide_s1_results_dict.items() if key in TIDE_TYPES
    }

    # loop search results
    for tide_type, results in tide_s1_results_dict.items():
        logger.info(f"Tide type: {tide_type}")
        logger.info(f"Total results: {len(results)}")

        # save results into geojson file
        result_path = suboutput_dir / f"{region_id:04d}_s1_{tide_type}.geojson"
        dag.serialize(results, result_path)

        output_paths = []
        rtc_output_paths = []
        for result in results:
            # get information based on individual result
            platform_id = result.properties["platform"]
            product_type = result.properties["productType"]
            sensor_mode = result.properties["sensorMode"]
            sensing_time = parse(result.properties["startTimeFromAscendingNode"])
            product_title = result.properties["title"]

            year = sensing_time.year
            month = sensing_time.month
            day = sensing_time.day
            sensing_time_str = sensing_time.strftime("%Y%m%dT%H%M%S")

            # define s1 directory based on result information
            s1_dir = S1_BASE_DIR.joinpath(
                platform_id.lower(),
                f"{platform_id}_{sensor_mode}_{product_type}".lower(),
                f"{year}",
                f"{month:02d}",
                f"{day:02d}",
            )

            # start to download eodag result
            dag.download(result, outputs_prefix=s1_dir, extract=False)