import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

OUTPUT_DIR = Path("/home/barata-serv/otomatisasi_barata/datasets/s1/s1_gamma0_rtc/coastal")
temp_dirs = sorted(OUTPUT_DIR.glob("*/temp"))

for i, temp_dir in enumerate(temp_dirs):
    logger.info(f"({i+1}/{len(temp_dirs)} {temp_dir} is deleting...)")
    shutil.rmtree(temp_dir)