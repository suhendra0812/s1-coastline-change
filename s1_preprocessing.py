# import builtin packages
import logging
from pathlib import Path
import shutil

# import third party packages
from dateutil.parser import parse
import geopandas as gpd
from sertit import misc, snap

def is_dim_img_exists(dim_path: Path, pattern: str = "*", n_img: int = 1) -> bool:
    """
    Fungsi untuk mengecek ada atau tidak file '.img' di dalam folder '.data'
    berdasarkan pattern dan jumlah file img yang diinginkan
    """
    data_dir = dim_path.parent / f"{dim_path.stem}.data" # mendefinisikan folder .data berdasarkan dim_path
    img_paths = sorted(data_dir.glob(f"{pattern}.img")) # mencari file .img berdasarkan pattern di dalam folder data_dir
    return len(img_paths) == n_img # mengembalikan nilai boolean dengan mengecek berdasarkan kesamaan jumlah file yang diiginkan

# setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# define region id
REGION_IDS = [671] # list region id yang dipilih
# define list of tide type
TIDE_TYPES = ["mean"] # list tipe pasut yang dipilih. Pilihannya bisa: ['mean', 'ht', 'lt']

# set working directory
WORK_DIR = Path(__file__).parent.resolve()
# define output directory
OUTPUT_DIR = WORK_DIR / "output"
# define s1 base directory
DATASET_DIR = Path("/home/barata-serv/otomatisasi_barata/datasets")
# define output dataset base directory
OUTPUT_DATASET_DIR = DATASET_DIR / "s1" / "s1_gamma0_rtc" / "coastal"

# define paths
REGION_PATH = WORK_DIR / "region" / "coastal_grids.geojson"
RTC_GRAPH_PATH = WORK_DIR / "graph" / "s1_rtc.xml"
COREG_GRAPH_PATH = WORK_DIR / "graph" / "coregistration.xml"
BS_GRAPH_PATH = WORK_DIR / "graph" / "band_select.xml"

# read region file as GeoDataFrame
region_gdf = gpd.read_file(REGION_PATH) # membaca file region sebagai objek GeoDataFrame

# loop list of region ids
for i, region_id in enumerate(REGION_IDS):
    # filter region
    selected_region = region_gdf.iloc[[region_id - 1]]

    # get geometry and wkt from geopandas object
    selected_region_geom = selected_region.geometry.unary_union # ambil informasi geometry dari objek GeoDataFrame
    selected_region_wkt = selected_region_geom.wkt # convert format geometry menjadi format wkt yang diinginkan SNAP

    # define suboutput directory
    suboutput_dir = OUTPUT_DIR / f"{region_id:04d}"
    suboutput_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Region ({i+1}/{len(selected_region)}) | ID: {region_id}")

    # loop tide type list
    for tide_type in TIDE_TYPES:
        logger.info(f"Tide type: {tide_type}")

        # read result path which has created from s1 downloader script
        result_path = suboutput_dir / f"{region_id:04d}_s1_{tide_type}.geojson"
        results = gpd.read_file(result_path) # membaca metadata hasil download sentinel-1
        results.sort_values(by="startTimeFromAscendingNode", ignore_index=True) # urutkan nilainya berdasarkan waktu perekamanan

        logger.info(f"Total s1 data: {len(results)}")

        s1_paths = [] # list kosong yang nanti akan diisi oleh path sentinel-1 hasil download
        output_paths = [] # list kosong yang nanti akan diisi oleh path output

        # loop results
        for i, result in results.iterrows():
            # ambil informasi dari metadata sentinel-1 untuk penamaan folder dan file
            platform_id = result["platform"] # hasilnya 'S1'
            product_type = result["productType"] # hasilnya 'GRD'
            sensor_mode = result["sensorMode"] # hasilnya 'IW'
            sensing_time = parse(result["startTimeFromAscendingNode"]) # hasilnya waktu sebagai objek datetime
            product_title = result["title"] # hasilnya nama file sentinel-1

            # ambil informasi tahun, bulan dan hari dari waktu perekaman
            year = sensing_time.year
            month = sensing_time.month
            day = sensing_time.day
            sensing_time_str = sensing_time.strftime("%Y%m%dT%H%M%S")

            # define s1 directory based on result information
            s1_dir = DATASET_DIR.joinpath(
                platform_id.lower(),
                f"{platform_id}_{sensor_mode}_{product_type}".lower(),
                f"{year}",
                f"{month:02d}",
                f"{day:02d}",
            )

            # define s1 path
            s1_path = s1_dir.joinpath(f"{product_title}.zip") # path sentinel-1 hasil download

            # define output name
            output_name = f"{region_id:04d}_{sensing_time_str}_{platform_id}_{sensor_mode}_{product_type}_gamma0-rtc_VH.tif" # nama output

            # define output path
            output_path = OUTPUT_DATASET_DIR.joinpath(f"{region_id:04d}", output_name) # output path yang nanti sebagai hasil akhir preprocessing
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # append path to use in next step
            s1_paths.append(s1_path) # menambahkan path sentinel-1 ke list path sentinel-1 di atas
            output_paths.append(output_path) # menambahkan output path ke list output path di atas

        # ============== SENTINEL-1 PREPROCESSING ======================

        if all([path.exists() for path in output_paths]): # mengecek jika semua output path sudah ada maka maka tidak ada dilakukan preprocessing lagi
            logger.info("All outputs are existed")
        else:
            # run the following command if any of output paths is not exists

            # initialize temp directory
            temp_dir = OUTPUT_DATASET_DIR.joinpath(f"{region_id:04d}", "temp") # membuat directory temp untuk menyimpan sementara hasil setiap langkah preprocessing
            temp_dir.mkdir(parents=True, exist_ok=True)

            logger.info("RTC Preprocessing...")
            for i, s1_path in enumerate(s1_paths): # looping list sentinel-1, 'enumerate' maksudnya untuk penomeran setiap elemen di dalam list mulai dari 0 sampai jumlah elemen yang ada di dalam list
                logger.info(f"({i+1}/{len(s1_paths)}) Sentinel-1 path: {s1_path}")

                # define rtc output path
                rtc_output_path = temp_dir / f"{s1_path.stem}_rtc.dim"

                logger.info(f"RTC output path: {rtc_output_path}")

                # mendapatkan command GPT berdasarkan parameter RTC
                # contoh bentuk lainnya: cmd = f"gpt {GRAPH_PATH} -Pinput={input_path} -Ppol={pol} -Psubset='{region}' -Poutput={output_path}"
                rtc_cmd = snap.get_gpt_cli(
                    RTC_GRAPH_PATH,
                    [
                        f"-Pinput={s1_path}",
                        f"-Ppol=VH",
                        f"-Psubset='{selected_region_wkt}'",
                        f"-Poutput={rtc_output_path}",
                    ],
                )

                if not is_dim_img_exists(rtc_output_path):
                    # run command line if output path is not existed
                    misc.run_cli(rtc_cmd) # menjalankan command GPT di terminal
                else:
                    logger.info("RTC output path is already existed")

            logger.info("Coregistration...")
            
            # list output rtc dan mengurutkannya secara menurun berdasarkan informasi waktu perekemanan di dalam nama file
            rtc_output_paths = sorted(
                temp_dir.glob("*rtc.dim"), # melakukan list nama file berdasarkan pattern yang diinginkan
                key=lambda x: parse(x.stem.split("_")[4]), # ini fungsi untuk mengambil informasi waktu prekemanan di dalam nama file
                reverse=True, # reverse itu maksudnya dibalik, kareka defaultnya itu urutannya meningkat
            )

            # ini hanya untuk mengeluarkan info di logging
            for i, path in enumerate(rtc_output_paths):
                if i == 0:
                    logger.info(f"{i+1}. {path.stem} (Master)")
                else:
                    logger.info(f"{i+1}. {path.stem}")

            # menggabungkan list output hasil rtc kedalam teks dengan pemisah ','
            # contoh: output_paths = [a, b, c, d] -> output_list = "a,b,c,d" 
            rtc_file_list = ",".join([str(path) for path in rtc_output_paths])

            # define coregistration output path
            coreg_output_path = (
                temp_dir / f"{rtc_output_paths[0].stem}_coreg.dim"
            )

            logger.info(f"Coregistration output path: {coreg_output_path}")

            # get gpt command line based on coregistration graph parameter
            coreg_cmd = snap.get_gpt_cli(
                COREG_GRAPH_PATH,
                [
                    f"-Pfilelist={rtc_file_list}",
                    f"-Poutput={coreg_output_path}",
                ],
            )

            if not is_dim_img_exists(coreg_output_path, n_img=len(rtc_output_paths)):
                # run command line if output path is already existed
                misc.run_cli(coreg_cmd)
            else:
                logger.info("Coregistration output path is already existed")

            # list coregistration output band and sort them based on date info in the filename
            coreg_band_paths = sorted(
                coreg_output_path.parent.joinpath(
                    f"{coreg_output_path.stem}.data"
                ).glob("*.img"),
                key=lambda x: parse(x.stem.split("_")[3]),
            )

            # loop through coregistration band path list
            for coreg_band_path, output_path in zip(coreg_band_paths, output_paths):
                logger.info("Band selection and convert to GeoTIFF...")

                # get the basename of the coregistration output band folder (*.data)
                # contoh:
                # coreg_band_path = ./S1B_IW_GRDH_1SDV_20211213T104954_20211213T105019_030005_039514_0420_rtc_coreg.data/Gamma0_VH_mst_13Dec2021.img
                # coreg_basename = S1B_IW_GRDH_1SDV_20211213T104954_20211213T105019_030005_039514_0420_rtc_coreg
                coreg_basename = coreg_band_path.parent.stem

                # get band name from coregistration band path (*.img)
                # contoh:
                # coreg_band_path = ./S1B_IW_GRDH_1SDV_20211213T104954_20211213T105019_030005_039514_0420_rtc_coreg.data/Gamma0_VH_mst_13Dec2021.img
                # coreg_bandname = Gamma0_VV_1Jan2022
                coreg_bandname = coreg_band_path.stem

                logger.info(f"Selecting band {coreg_bandname}...")

                # define band select output path
                bs_output_path = (
                    temp_dir
                    / f"{coreg_basename}_VH.tif"
                )

                # get gpt command line based on band select graph parameter
                bs_cmd = snap.get_gpt_cli(
                    BS_GRAPH_PATH,
                    [
                        f"-Pinput={coreg_output_path}",
                        f"-Pbandname={coreg_bandname}",
                        f"-Poutput={output_path}",
                    ],
                )

                if not output_path.exists():
                    # run command line if output path is already existed
                    misc.run_cli(bs_cmd)

                logger.info(f"Saved to {output_path}")

        # remove temp directory recursively
        # shutil.rmtree(temp_dir, ignore_errors=True)

        # ============== DATASET METADATA GENERATING ======================
        try:
            from eodatasets3 import DatasetPrepare # type: ignore

            for output_path in output_paths:
                # define metadata path
                metadata_path = output_path.with_suffix(".odc-metadata.yaml")

                logger.info(f"Metadata path: {metadata_path}")

                if not metadata_path.exists():
                    # get sensing datetime from filename
                    sensing_time = parse(output_path.stem.split("_")[1])

                    # prepare dataset metadata
                    with DatasetPrepare(
                        collection_location=output_path.parent,
                        dataset_location=output_path.parent,
                        metadata_path=metadata_path,
                    ) as p:
                        p.platform = "sentinel-1"
                        p.product_family = "gamma0_rtc"
                        p.names.platform_abbreviated = platform_id.lower() # contoh: s1
                        p.datetime = sensing_time
                        p.properties.update({"odc:file_format": "GeoTIFF"})
                        product_name = f"{platform_id.lower()}_{sensor_mode.lower()}_{p.product_family}_vh" # contoh: s1_iw_grd_vh
                        p.names.product_name = product_name
                        p.processed_now()

                        p.note_measurement(
                            "vh", output_path.name, relative_to_dataset_location=True
                        )

                        p.done()

                assert metadata_path.exists()
                logger.info("Metadata created")

        except ImportError as e:
            logger.warning(f"{e}. Metadata creation is not executed.")
