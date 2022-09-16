import logging
import os
from pathlib import Path

from qgis.core import ( # type: ignore
    QgsApplication,
    QgsProject,
    QgsVectorLayer,
    QgsRasterLayer,
    QgsRectangle,
)
from qgis.gui import QgsMapCanvas # type: ignore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

QgsApplication.setPrefixPath("/usr", True)
qgis_app = QgsApplication([], False)

qgis_app.initQgis()

project_path = "./layout/s1_coastline_change.qgz"

project = QgsProject.instance()
project.read(project_path)

coastline_group = project.layerTreeRoot().findGroup("coastlines")
transect_analysis_group = project.layerTreeRoot().findGroup("transect_analysis")
image_group = project.layerTreeRoot().findGroup("images")

coastline_group.removeAllChildren()
transect_analysis_group.removeAllChildren()
image_group.removeAllChildren()

coastline_paths = sorted(Path("./output").glob("*/*coastlines.geojson"))
transect_analysis_paths = sorted(Path("./output").glob("*/*transect_analysis.geojson"))
image_paths = sorted(Path("./output").glob("*/*coreg.tif"))

coastline_layer_template_path = "./layout/coastline_layer_template.qml"
transect_analysis_layer_template_path = "./layout/transect_analysis_layer_template.qml"

logger.info("Load coastlines and transect analysis")
for i, (coastline_path, transect_analysis_path) in enumerate(
    zip(coastline_paths, transect_analysis_paths)
):
    logger.info(f"({i+1}/{len(coastline_paths)}) Coastline: {coastline_path}")
    logger.info(
        f"({i+1}/{len(coastline_paths)}) Transect analysis: {transect_analysis_path}"
    )

    coastline_layer = QgsVectorLayer(str(coastline_path), coastline_path.stem)
    transect_analysis_layer = QgsVectorLayer(
        str(transect_analysis_path), transect_analysis_path.stem
    )

    coastline_layer.loadNamedStyle(coastline_layer_template_path)
    transect_analysis_layer.loadNamedStyle(transect_analysis_layer_template_path)

    project.addMapLayers([coastline_layer, transect_analysis_layer], False)

    coastline_group.addLayer(coastline_layer)
    transect_analysis_group.addLayer(transect_analysis_layer)

extent = QgsRectangle()
extent.setMinimal()

logger.info("Load images")
for i, image_path in enumerate(image_paths):
    logger.info(f"({i+1}/{len(image_paths)}) Image: {image_path}")
    region_id = image_path.parent.name
    region_group = image_group.findGroup(region_id)
    if not region_group:
        region_group = image_group.addGroup(region_id)

    image_layer = QgsRasterLayer(str(image_path), image_path.stem)
    project.addMapLayer(image_layer, False)
    region_group.addLayer(image_layer)

    extent.combineExtentWith(image_layer.extent())

QgsMapCanvas().setExtent(extent)
QgsMapCanvas().refresh()

project.write(project_path)

xmin = extent.xMinimum()
xmax = extent.xMaximum()
ymin = extent.yMinimum()
ymax = extent.yMaximum()

qgis_app.exitQgis()

os.system(f"qgis.bin --project {project_path} --extent {xmin},{ymin},{xmax},{ymax}")
