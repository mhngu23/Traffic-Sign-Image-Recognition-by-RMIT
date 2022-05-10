from datasetReader import DatasetReader

if __name__ == '__main__':
    image_folder_path = "..\\trafficsigns_dataset"
    imageDataReader = DatasetReader(image_folder_path)
    imageDataReader.create_shape_dataset()
    print(len(imageDataReader.shapeImageFeatures))
    print(len(imageDataReader.shapeClassName))
