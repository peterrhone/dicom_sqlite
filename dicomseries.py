
#  make a class for each series, which stores info about it, e.g. the series name, the files sorted by image, contour, registration
class DicomSeries:
    def __init__(self, series_id):
        self.series_id = series_id
        self.num_slices = 0
        self.contour_files = []
        self.registration_files = []
        self.image_id_dict = {}
        self.RescaleSlope = 1.0
        self.RescaleIntercept = 0.0

    def add_image_file(self, image_file, img_ID):
        self.num_slices += 1
        self.image_id_dict[img_ID] = image_file

    def add_contour_file(self, contour_file):
        self.contour_files.append(contour_file)

    def add_registration_file(self, registration_file):
        self.registration_files.append(registration_file)

    def set_rescale_slope(self, slope):
        self.RescaleSlope = slope
    
    def set_rescale_intercept(self, intercept):
        self.RescaleIntercept = intercept
    
    def get_series_id(self):
        return self.series_id

    def get_image_files(self):
        return self.image_id_dict.values()

    def get_contour_files(self):
        return self.contour_files

    def get_registration_files(self):
        return self.registration_files

    def get_num_image_files(self):
        return len(self.image_id_dict)

    def get_num_contour_files(self):
        return len(self.contour_files)

    def get_num_registration_files(self):
        return len(self.registration_files)

    def get_image_file(self, id):
        return self.image_id_dict[id]

    def get_contour_file(self, i):
        return self.contour_files[i]

    def get_registration_file(self, i):
        return self.registration_files[i]

    def get_info(self):
        info_str = f"""Series ID: {self.series_id}
        Image files:        {self.get_num_image_files()}
        Contour files:      {self.get_num_contour_files()}
        Registration files: {self.get_num_registration_files()}"""
        return info_str