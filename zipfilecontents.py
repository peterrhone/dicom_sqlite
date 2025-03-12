import zipfile
# from os import walk
from dicomseries import DicomSeries
import pydicom as dicom

class ZipFileContents:
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename
        self.zip_ref = zipfile.ZipFile(self.path+filename, 'r')
        self.filenames = self.zip_ref.namelist()
        self.dicom_files = [name for name in self.filenames if name.endswith('.dcm')]
        self.contains_dicom_files = self.dicom_files != []
        self.series_dict = {}   #the series_id is the key and a DicomSeries object the value
        self.__get_all_series_from_zip()

    def __del__(self):
        self.zip_ref.close()

    def get_patient_id(self):
        pat_id = self.filename.split('.')[0]
        return pat_id
    
    def get_zip_file_contents(self):
        return self.zip_ref.namelist()

    def get_contour_data(self, series):
        if series.get_num_contour_files() != 1:
            print(f"ZipFileContents.get_contour_data: There should be only one contour file per series. Found {series.get_num_contour_files()} files.")
            return None
        else:
            f=self.zip_ref.open(series.get_contour_files()[0]) 
        # return dicom.read_file(f)
        return dicom.dcmread(f)

    def get_image_given_img_ID(self, series, img_ID):
        try:
            dcm = series.get_image_file(img_ID)
        except KeyError:
            print(f"ZipFileContents.get_image_given_img_ID: Image with ID {img_ID} not found.")
            return None
        # return dicom.read_file(self.zip_ref.open(dcm))
        return dicom.dcmread(self.zip_ref.open(dcm))
    
    def get_image_given_filename(self, fname):
        # return dicom.read_file(self.zip_ref.open(fname))
        return dicom.dcmread(self.zip_ref.open(fname))

    def __get_all_series_from_zip(self):
        tmptype = ""
        if not self.contains_dicom_files:
            print("ZipFileContents:_get_all_series_from_zip(): No dicom files found in the zip file.")
            self.contains_dicom_files = False
            return None
        for dcm in self.dicom_files:
            # f = dicom.read_file(self.zip_ref.open(dcm))
            f = dicom.dcmread(self.zip_ref.open(dcm))
            #determine the content type of the dicom file
            match f.SOPClassUID.name:
                case "CT Image Storage":
                    s_name = f.SeriesInstanceUID
                    tmptype = "CT"
                case "RT Structure Set Storage":
                    s_name = f.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
                    tmptype = "RT"
                case "Spatial Registration Storage":
                    s_name = f.ReferencedSeriesSequence[0].SeriesInstanceUID
                    tmptype = "REG"
                case _:
                    print(f"ZipFileContents:_get_all_series_from_zip(): Unknown SOPClassUID: {f.SOPClassUID.name}")
        
            #acreate a new series object if it does not exist and add it to the series dictionary
            if s_name not in self.series_dict:
                self.series_dict[s_name] = DicomSeries(s_name)
            
            #add the dicom file to the series object
            match tmptype:
                case "CT":
                    id = f.SOPInstanceUID
                    # instace_num = f.InstanceNumber
                    self.series_dict[s_name].add_image_file(dcm, id)
                case "RT":
                    self.series_dict[s_name].add_contour_file(dcm)
                case "REG":
                    self.series_dict[s_name].add_registration_file(dcm)

        #sort CT images by slice (=instance number)
        self.__sort_series_by_instance_number()
    
    def __sort_series_by_instance_number(self):
        for series in self.series_dict:
            #sort the dictionary by the instance number
            self.series_dict[series].image_id_dict = dict(sorted(\
                self.series_dict[series].image_id_dict.items(), \
                    key=lambda item: dicom.dcmread(\
                        self.zip_ref.open(item[1])).InstanceNumber))
            
    def get_all_series(self):
        return self.series_dict.values()