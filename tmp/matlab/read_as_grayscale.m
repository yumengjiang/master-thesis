function img = read_as_grayscale(path_to_file)
	img = read_image(path_to_file);
    img=mean(img,3);
end