from . import open_image as oi
from . import array_trs as art
from . import create as cr
from osgeo import gdal,gdal_array
    
def getSHP(img_path,img_name,save_path,extend_name,result_array):
    img=oi.open_tiff(img_path,img_name)
    H,W=img[1],img[2]
    result_array=result_array.reshape(H,W)
    dats = cr.create_tiff(nb_channels=1,new_tiff_name="outlier_image_"+extend_name+img_name+".tif",width = W, \
            height= H,data_array=result_array,datatype=gdal.GDT_UInt16, \
            geotransformation=img[4],projection=img[5])

    #use the ds_tiff to get an outlier image file for QIGS
    #use .shp file to point out the outliers
    cr.vectorize_tiff(main_path=save_path,\
            shp_name="outlier_image_"+extend_name+img_name,ds_tiff=dats)
    dats=None

