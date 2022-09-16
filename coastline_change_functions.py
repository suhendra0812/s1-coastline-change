import copy
import logging
from typing import Any, Dict

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import registration
import xarray as xr
from pystac import Item
from scipy import ndimage
from shapely.geometry import LineString, MultiLineString, shape
from skimage import filters, measure, morphology

logger = logging.getLogger("coastline_change_functions")


def intersection_percent(item: Item, aoi: Dict[str, Any]) -> float:
    """The percentage that the Item's geometry intersects the AOI. An Item that
    completely covers the AOI has a value of 100.
    """
    geom_item = shape(item.geometry)
    geom_aoi = shape(aoi)

    intersected_geom = geom_aoi.intersection(geom_item)

    intersection_percent = (intersected_geom.area * 100) / geom_aoi.area

    return intersection_percent


def filter_tide(tide_group, tide):
    tide_list = [
        xr.concat(sorted(group, key=lambda x: abs(x.tide - tide))[:1], dim="time")
        for _, group in tide_group
    ]
    tide_data = xr.concat(tide_list, dim="time")
    return tide_data


def get_gradient(src):
    """Method which calculates and returns the gradients for the input image, which are
    better suited for co-registration

    :param src: input image
    :type src: ndarray
    :return: ndarray
    """
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def register(src, trg, trg_mask=None, src_mask=None):
    """Implementation of pair-wise registration using thunder-registration

    For more information on the model estimation, refer to https://github.com/thunder-project/thunder-registration
    This function takes two 2D single channel images and estimates a 2D translation that best aligns the pair. The
    estimation is done by maximising the correlation of the Fourier transforms of the images. Once, the translation
    is estimated, it is applied to the (multichannel) image to warp and, possibly, ot hte ground-truth. Different
    interpolations schemes could be more suitable for images and ground-truth values (or masks).

    :param src: 2D single channel source moving image
    :param trg: 2D single channel target reference image
    :param src_mask: Mask of source image. Not used in this method.
    :param trg_mask: Mask of target image. Not used in this method.
    :return: Estimated 2D transformation matrix of shape 2x3
    """
    # Initialise instance of CrossCorr object
    ccreg = registration.CrossCorr()
    # padding_value = 0
    # Compute translation between a pair of images
    model = ccreg.fit(src, reference=trg)
    # Get translation as an array
    translation = [-x for x in model.toarray().tolist()[0]]
    # Fill in transformation matrix
    warp_matrix = np.eye(2, 3)
    warp_matrix[0, 2] = translation[1]
    warp_matrix[1, 2] = translation[0]
    # Return transformation matrix
    return warp_matrix


def is_registration_suspicious(warp_matrix):
    """Static method that checks if estimated linear transformation could be implausible.

    This function checks whether the norm of the estimated translation or the rotation angle exceed predefined
    values. For the translation, a maximum translation radius of 20 pixels is flagged, while larger rotations than
    20 degrees are flagged.

    :param warp_matrix: Input linear transformation matrix
    :type warp_matrix: ndarray
    :return: 0 if registration doesn't exceed threshold, 1 otherwise
    """
    MAX_TRANSLATION = 20
    MAX_ROTATION = np.pi / 9

    if warp_matrix is None:
        return 1

    cos_theta = np.trace(warp_matrix[:2, :2]) / 2
    rot_angle = np.arccos(cos_theta)
    transl_norm = np.linalg.norm(warp_matrix[:, 2])
    return (
        1 if int((rot_angle > MAX_ROTATION) or (transl_norm > MAX_TRANSLATION)) else 0
    )


def warp(warp_matrix, img, iflag=cv2.INTER_NEAREST):
    """Function to warp input image given an estimated 2D linear transformation

    :param warp_matrix: Linear 2x3 matrix to use to linearly warp the input images
    :type warp_matrix: ndarray
    :param img: Image to be warped with estimated transformation
    :type img: ndarray
    :param iflag: Interpolation flag, specified interpolation using during resampling of warped image
    :type iflag: cv2.INTER_*
    :return: Warped image using the linear matrix
    """

    height, width = img.shape[:2]
    warped_img = np.zeros_like(img, dtype=img.dtype)

    iflag += cv2.WARP_INVERSE_MAP

    # Check if image to warp is 2D or 3D. If 3D need to loop over channels
    if img.ndim == 2:
        warped_img = cv2.warpAffine(
            img.astype(np.float32), warp_matrix, (width, height), flags=iflag
        ).astype(img.dtype)

    elif img.ndim == 3:
        for idx in range(img.shape[-1]):
            warped_img[..., idx] = cv2.warpAffine(
                img[..., idx].astype(np.float32),
                warp_matrix,
                (width, height),
                flags=iflag,
            ).astype(img.dtype)
    else:
        raise ValueError(f"Image has incorrect number of dimensions: {img.ndim}")

    return warped_img


def coregistration(data: xr.DataArray) -> xr.DataArray:
    new_data = copy.deepcopy(data)
    new_data = new_data.astype(np.float32)
    time_len = len(new_data.time.values)
    sliced_data = new_data.values
    sliced_data = [get_gradient(d) for d in sliced_data]
    sliced_data = [
        np.clip(d, np.percentile(d, 5), np.percentile(d, 95)) for d in sliced_data
    ]

    for idx in range(time_len - 1, 0, -1):
        src_mask, trg_mask = None, None
        warp_matrix = register(
            sliced_data[idx], sliced_data[idx - 1], src_mask=src_mask, trg_mask=trg_mask
        )
        rflag = is_registration_suspicious(warp_matrix)
        if rflag:
            warp_matrix = np.eye(2, 3)

        new_data[idx - 1] = warp(warp_matrix=warp_matrix, img=new_data[idx - 1].values)
        sliced_data[idx - 1] = warp(warp_matrix, sliced_data[idx - 1])
    return new_data


def db_scale(img):
    db_output = 10 * np.log10(img, where=img > 0)
    return db_output


def lee_filter(img, size):
    img_mean = ndimage.uniform_filter(img, size)
    img_sqr_mean = ndimage.uniform_filter(img**2, size)
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = ndimage.variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def segmentation(img: np.ndarray, threshold: float = None) -> np.ndarray:
    img = np.where(~np.isnan(img), img, 0) # jika ada nilai NaN di dalam piksel citra, maka diisi angka 0. karena akan error pada perhitungan threshold menggunakan otsu
    if not threshold: # jika nilai threshold tidak disediakan, maka dilakukan perhitungan nilai threshold otomatis menggunakan Otsu
        threshold = filters.threshold_otsu(img)
    logger.info(f"Threshold: {threshold}")
    binary = (img > threshold).astype(np.uint8) # proses thresholding citra berdasarkan nilai threshold, kemudian diubah ke dalam angka 0 dan 1
    binary = ndimage.binary_fill_holes(binary) # pengisian lubang-lubang di dalam area piksel 1 (putih)
    binary = morphology.remove_small_objects(binary).astype(int) # penghapusan objek-objek kecil (piksel 1/putih)
    binary = morphology.closing(binary, morphology.disk(5)) # untuk smoothing area garis pantai
    return binary.astype(np.uint8)


def subpixel_contours(
    da,
    z_values=[0.0],
    crs=None,
    affine=None,
    attribute_df=None,
    output_path=None,
    min_vertices=2,
    dim="time",
    errors="ignore",
    verbose=False,
):
    def contours_to_multiline(da_i, z_value, min_vertices=2):
        """
        Helper function to apply marching squares contour extraction
        to an array and return a data as a shapely MultiLineString.
        The `min_vertices` parameter allows you to drop small contours
        with less than X vertices.
        """

        # Extracts contours from array, and converts each discrete
        # contour into a Shapely LineString feature. If the function
        # returns a KeyError, this may be due to an unresolved issue in
        # scikit-image: https://github.com/scikit-image/scikit-image/issues/4830
        line_features = [
            LineString(i[:, [1, 0]])
            for i in measure.find_contours(da_i.data, z_value)
            if i.shape[0] > min_vertices
        ]

        # Output resulting lines into a single combined MultiLineString
        return MultiLineString(line_features)

    # Check if CRS is provided as a xarray.DataArray attribute.
    # If not, require supplied CRS
    try:
        crs = da.crs
    except:
        if crs is None:
            raise ValueError(
                "Please add a `crs` attribute to the "
                "xarray.DataArray, or provide a CRS using the "
                "function's `crs` parameter (e.g. 'EPSG:3577')"
            )

    # Check if Affine transform is provided as a xarray.DataArray method.
    # If not, require supplied Affine
    try:
        affine = da.geobox.transform
    except KeyError:
        affine = da.transform
    except:
        if affine is None:
            raise TypeError(
                "Please provide an Affine object using the "
                "`affine` parameter (e.g. `from affine import "
                "Affine; Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "
                "6886890.0)`"
            )

    # If z_values is supplied is not a list, convert to list:
    z_values = (
        z_values
        if (isinstance(z_values, list) or isinstance(z_values, np.ndarray))
        else [z_values]
    )

    # Test number of dimensions in supplied data array
    if len(da.shape) == 2:
        if verbose:
            logger.info(f"Operating in multiple z-value, single array mode")
        dim = "z_value"
        contour_arrays = {
            str(i)[0:10]: contours_to_multiline(da, i, min_vertices) for i in z_values
        }

    else:

        # Test if only a single z-value is given when operating in
        # single z-value, multiple arrays mode
        if verbose:
            logger.info(f"Operating in single z-value, multiple arrays mode")
        if len(z_values) > 1:
            raise ValueError(
                "Please provide a single z-value when operating "
                "in single z-value, multiple arrays mode"
            )

        contour_arrays = {
            str(i)[0:10]: contours_to_multiline(da_i, z_values[0], min_vertices)
            for i, da_i in da.groupby(dim)
        }

    # If attributes are provided, add the contour keys to that dataframe
    if attribute_df is not None:

        try:
            attribute_df.insert(0, dim, contour_arrays.keys())
        except ValueError:

            raise ValueError(
                "One of the following issues occured:\n\n"
                "1) `attribute_df` contains a different number of "
                "rows than the number of supplied `z_values` ("
                "'multiple z-value, single array mode')\n"
                "2) `attribute_df` contains a different number of "
                "rows than the number of arrays along the `dim` "
                "dimension ('single z-value, multiple arrays mode')"
            )

    # Otherwise, use the contour keys as the only main attributes
    else:
        attribute_df = list(contour_arrays.keys())

    # Convert output contours to a geopandas.GeoDataFrame
    contours_gdf = gpd.GeoDataFrame(
        data=attribute_df, geometry=list(contour_arrays.values()), crs=crs
    )

    # Define affine and use to convert array coords to geographic coords.
    # We need to add 0.5 x pixel size to the x and y to obtain the centre
    # point of our pixels, rather than the top-left corner
    shapely_affine = [
        affine.a,
        affine.b,
        affine.d,
        affine.e,
        affine.xoff + affine.a / 2.0,
        affine.yoff + affine.e / 2.0,
    ]
    contours_gdf["geometry"] = contours_gdf.affine_transform(shapely_affine)

    # Rename the data column to match the dimension
    contours_gdf = contours_gdf.rename({0: dim}, axis=1)

    # Drop empty timesteps
    empty_contours = contours_gdf.geometry.is_empty
    failed = ", ".join(map(str, contours_gdf[empty_contours][dim].to_list()))
    contours_gdf = contours_gdf[~empty_contours]

    # Raise exception if no data is returned, or if any contours fail
    # when `errors='raise'. Otherwise, logger.info failed contours
    if empty_contours.all() and errors == "raise":
        raise RuntimeError(
            "Failed to generate any valid contours; verify that "
            "values passed to `z_values` are valid and present "
            "in `da`"
        )
    elif empty_contours.all() and errors == "ignore":
        if verbose:
            logger.info(
                "Failed to generate any valid contours; verify that "
                "values passed to `z_values` are valid and present "
                "in `da`"
            )
    elif empty_contours.any() and errors == "raise":
        raise Exception(f"Failed to generate contours: {failed}")
    elif empty_contours.any() and errors == "ignore":
        if verbose:
            logger.info(f"Failed to generate contours: {failed}")

    # If asked to write out file, test if geojson or shapefile
    if output_path and output_path.endswith(".geojson"):
        if verbose:
            logger.info(f"Writing contours to {output_path}")
        contours_gdf.to_crs("EPSG:4326").to_file(filename=output_path, driver="GeoJSON")

    if output_path and output_path.endswith(".shp"):
        if verbose:
            logger.info(f"Writing contours to {output_path}")
        contours_gdf.to_file(filename=output_path)

    return contours_gdf


def smooth_linestring(linestring, smooth_sigma):
    """
    Uses a gauss filter to smooth out the LineString coordinates.
    """
    smooth_x = np.array(ndimage.gaussian_filter1d(linestring.xy[0], smooth_sigma))
    smooth_y = np.array(ndimage.gaussian_filter1d(linestring.xy[1], smooth_sigma))
    smoothed_coords = np.hstack((smooth_x, smooth_y))
    smoothed_coords = zip(smooth_x, smooth_y)
    linestring_smoothed = LineString(smoothed_coords)
    return linestring_smoothed


def create_transects(line, space, length, crs):
    # Profile spacing. The distance at which to space the perpendicular profiles
    # In the same units as the original shapefile (e.g. metres)
    space = space

    # Length of cross-sections to calculate either side of central line
    # i.e. the total length will be twice the value entered here.
    # In the same co-ordinates as the original shapefile
    length = length

    # Define a schema for the output features. Add a new field called 'Dist'
    # to uniquely identify each profile

    transect_list = []

    # Calculate the number of profiles to generate
    n_prof = int(line.length / space)

    # Start iterating along the line
    for prof in range(1, n_prof + 1):
        # Get the start, mid and end points for this segment
        seg_st = line.interpolate((prof - 1) * space)
        seg_mid = line.interpolate((prof - 0.5) * space)
        seg_end = line.interpolate(prof * space)

        # Get a displacement vector for this segment
        vec = np.array(
            [
                [
                    seg_end.x - seg_st.x,
                ],
                [
                    seg_end.y - seg_st.y,
                ],
            ]
        )

        # Rotate the vector 90 deg clockwise and 90 deg counter clockwise
        rot_anti = np.array([[0, -1], [1, 0]])
        rot_clock = np.array([[0, 1], [-1, 0]])
        vec_anti = np.dot(rot_anti, vec)
        vec_clock = np.dot(rot_clock, vec)

        # Normalise the perpendicular vectors
        len_anti = ((vec_anti**2).sum()) ** 0.5
        vec_anti = vec_anti / len_anti
        len_clock = ((vec_clock**2).sum()) ** 0.5
        vec_clock = vec_clock / len_clock

        # Scale them up to the profile length
        vec_anti = vec_anti * length
        vec_clock = vec_clock * length

        # Calculate displacements from midpoint
        prof_st = (seg_mid.x + float(vec_clock[0]), seg_mid.y + float(vec_clock[1]))
        prof_end = (seg_mid.x + float(vec_anti[0]), seg_mid.y + float(vec_anti[1]))

        distance = (prof - 0.5) * space
        transect = LineString([prof_end, prof_st])

        gdf = gpd.GeoDataFrame({"distance": [distance]}, geometry=[transect])

        transect_list.append(gdf)

    transect_gdf = pd.concat(transect_list, ignore_index=True)
    transect_gdf.crs = crs

    return transect_gdf


def transect_analysis(line_gdf, transect_gdf, time_column, reverse=False):
    line_gdf = line_gdf.copy()
    transect_gdf = transect_gdf.copy()

    line_gdf[time_column] = pd.to_datetime(line_gdf[time_column]) # memastikan formatnya menjadi objek datetime python
    # line_gdf["time_idx"], _ = pd.factorize(line_gdf[time_column])

    line_gdf.sort_values(by=time_column, inplace=True, ignore_index=True) # diurutkan berdasarkan waktu
    transect_gdf.reset_index(drop=True, inplace=True)

    analysis_list = [] # list hasil analisis transek

    for i, transect in transect_gdf.iterrows(): # looping row GeoDataFrame transek
        start, end = transect.geometry.boundary.geoms # mengambil informasi titik awal dan akhir dari satu transek
        if reverse: # jika reverse titiknya ditukar
            start, end = end, start
        if any(line_gdf.geometry.intersects(transect.geometry)): # jika antara garis pantai dan transek ada yang bersentuhan, maka lakukan proses di bawah ini
            intersect_gdf = line_gdf.copy()
            # mengambil titik intersection-nya
            intersect_gdf.geometry = intersect_gdf.geometry.intersection(
                transect.geometry
            )
            geom_types = [geom.geom_type for geom in intersect_gdf.geometry] # list jenis geometry
            if geom_types.count("Point") == len(intersect_gdf): # jika jumlah jenis geometry 'Point' sama dengan jumlah garis pantai, amak lanjutkan ke proses di bawah ini
                analysis_data = {"name": [i]} # mengumpulkan infromasi hasil analisis, diawali dengan nama yang disi oleh urutan trasek
                
                # di bawah ini akan dilakukan perhitungan perubahan garis pantai setiap dua tahun sekali
                # contoh:
                # jika tahun di garis pantai = [2015, 2016, 2017, 2018], maka perubahan garis pantai akan dihitung sbb:
                # 1. Garis pantai 2015 dikurangi garis pantai 2016
                # 2. Garis pantai 2016 dikurangi garis pantai 2015
                # 3. Garis pantai 2017 dikurangi garis pantai 2016
                # 4. Garis pantai 2018 dikurangi garis pantai 2017
                
                # ini penjelasan kodingannya
                # looping jumlah garis pantai dan urutannya
                # jika jumlah garis pantai ada 3 berarti nilai j = [0, 1, 2] untuk digunakan sebagai indeks garis pantai lamam
                # nilai k = j + 1, maka hasilnya k = [1, 2, 3] untuk digunakan sebagai indeks garis pantai baru
                # kemudian dilakukan pengecekan, jika k (indeks garis pantai baru) terakhir sama dengan jumlah garis pantai maka proses berhenti (break)
                for j in range(len(intersect_gdf)):
                    k = j + 1
                    if k == len(intersect_gdf):
                        break
                    
                    oldest_intersect = intersect_gdf.iloc[j] # garis pantai tanggal lama
                    oldest_date = oldest_intersect[time_column] # tanggal garis pantai tanggal lama
                    oldest_geom = oldest_intersect.geometry # geometry garis pantai tanggal lama
                    oldest_distance = oldest_geom.distance(start) # jarak garis pantai tanggal lama dengan titik start dari transek

                    latest_intersect = intersect_gdf.iloc[k] # garis pantai tanggal baru
                    latest_date = latest_intersect[time_column] # garis pantai tanggal baru
                    latest_geom = latest_intersect.geometry # geometry garis pantai tanggal baru
                    latest_distance = latest_geom.distance(start) # jarak garis pantai tanggal baru dengan titik start dari transek

                    date_str = oldest_date.strftime("%Y%m%d") # ini hanya untuk penamaan kolom, setiap hasil analisis di kolom tabel akan ditambahkan tanggal garis pantainya
                    # date_idx = oldest_intersect["time_idx"]

                    if j > 0: # pengecekan jika j > 0 maka lakukan perhitungan perubahan di bawah ini
                        change = latest_distance - oldest_distance
                        rate = change / ((latest_date - oldest_date).days / 365)
                    else: # j = 0 maka belum ada perubahan
                        change = 0
                        rate = 0

                    # masukkan hasil analisis berdasarkan tanggal garis pantai
                    analysis_data[f"distance_{date_str}"] = [oldest_distance]
                    analysis_data[f"change_{date_str}"] = [change]
                    analysis_data[f"rate_{date_str}"] = [rate]

                # hasil analisis perubahan garis pantai disimpan dalam GeoDataFrame
                # kolom GeoDataFrame berisi:
                # - name (urutan transek)
                # - distance_[tanggal_garis_pantai] (jarak dari titk awal transek)
                # - change_[tanggal_garis_pantai] (perubahan jarak antara garis pantai lama dan baru)
                # - rate_[tanggal_garis_pantai] (laju perubahan jarak antara garis pantai lama dan baru per satuan waktu)
                # - mean_distance (rata-rata perubahan dari semua distance per tanggal)
                # - mean_change (rata-rata perubahan dari semua change per tanggal)
                # - mean_rate (rata-rata perubahan dari semua rate per tanggal)
                analysis_geom = LineString(intersect_gdf.geometry) # membuat objek geometry line/garis berdasarkan geometry dari titik-titik intersection
                analysis_gdf = gpd.GeoDataFrame(analysis_data, geometry=[analysis_geom]) # membuat objek GeoDataFrame berdasarkan hasil analisis dan geometry garis intersection

                # di bawah ini perhitungan rata-rata masing-masing kolom distance, change dan rate
                distance_columns = analysis_gdf.columns[
                    analysis_gdf.columns.str.contains("distance")
                ]
                analysis_gdf["mean_distance"] = analysis_gdf[distance_columns].mean(
                    axis=1
                )

                change_columns = analysis_gdf.columns[
                    analysis_gdf.columns.str.contains("change")
                ]
                analysis_gdf["mean_change"] = analysis_gdf[change_columns].mean(axis=1)

                rate_columns = analysis_gdf.columns[
                    analysis_gdf.columns.str.contains("rate")
                ]
                analysis_gdf["mean_rate"] = analysis_gdf[rate_columns].mean(axis=1)

                # hasil analisis setiap transek digabungkan ke dalam list
                analysis_list.append(analysis_gdf)

    if not analysis_list:
        logger.warning("No analysis resulted")
        return

    # penggabungan list hasil analisis menjadi satu objek GeoDataFrame
    transect_analysis_gdf = pd.concat(analysis_list, ignore_index=True)
    transect_analysis_gdf.crs = line_gdf.crs # sistem proyeksi disamakan dengan garis pantai

    return transect_analysis_gdf


def rescale(
    img: np.ndarray, target_type_min: float, target_type_max: float, target_type: type
) -> np.ndarray:
    img = np.clip(img, np.percentile(img, 5), np.percentile(img, 95))

    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax

    new_img = (a * img + b).astype(target_type)

    return new_img
