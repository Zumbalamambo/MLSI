from osgeo import gdal, gdal_array


def open_tiff(path, name):
    ds = gdal.Open(path+"/"+name+".tif")#open an tiff image
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    bands_nb = ds.RasterCount
    W = ds.RasterXSize
    H = ds.RasterYSize
    #the function to vectorize the image
    try:
        image_array = gdal_array.LoadFile(path + "/" + name+".tif")
    except:
        image_array = gdal_array.LoadFile(path + name+".tif")
    #close the raster dataset
    ds = None
    return image_array, H, W, bands_nb, geo, proj






