function img = read_image(path_to_file)
    img=imread(path_to_file);
    img=im2double(img);
end

