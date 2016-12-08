#include "ImageData.h"

using namespace std;

// Read the image from the given file, or set the error bit if the image
// could not be read.
ImageData::ImageData(string file, int target_size, bool boundingBox) {
  IplImage* gray_img;
  gray_img = cvLoadImage(file.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
  if (!gray_img) {
    readError = true;
    return;
  }

  // Normalize the image by first converting to binary and then resizing the
  // contentful part to a standard size.
  IplImage* bin_img = cvCreateImage(cvGetSize(gray_img), IPL_DEPTH_8U, 1);
  cvThreshold(gray_img, bin_img, 128, 255,
              CV_THRESH_BINARY | CV_THRESH_OTSU);

  if (boundingBox) {
    // Compute a bounding box around the image so that we may resize only the
    // contentful portion.
    int min_x = gray_img->width, max_x = 0;
    int min_y = gray_img->height, max_y = 0;
    for (int i = 0; i < gray_img->height; i++) {
      for (int j = 0; j < gray_img->width; j++) {
        // Black pixels have 0 value.
        if (cvGet2D(bin_img, i, j).val[0] == 0) {
          min_x = min(min_x, j);
          min_y = min(min_y, i);
          max_x = max(max_x, j);
          max_y = max(max_y, i);
        }
      }
    }

    cvSetImageROI(bin_img, cvRect(min_x, min_y, max_x, max_y));
  } else {
    cvSetImageROI(bin_img, cvRect(0, 0, gray_img->width, gray_img->height));
  }

  // cvGetSize() returns the size of the ROI.
  IplImage* big_img;
  big_img = cvCreateImage(cvGetSize(bin_img),
                      bin_img->depth,
                      bin_img->nChannels);
  cvCopy(bin_img, big_img, NULL);
  img = cvCreateImage(cvSize(target_size, target_size),
                      bin_img->depth,
                      bin_img->nChannels);
  cvResize(big_img, img);

  width = target_size;
  height = target_size;

  pixels = (float *)malloc(sizeof(float) * img->width * img->height);
  for (int i = 0; i < img->height; i++) {
    for (int j = 0; j < img->width; j++) {
      if (cvGet2D(img, i, j).val[0] == 0) {
        pixels[i * img->width + j] = 5.0f;
      } else {
        pixels[i * img->width + j] = -5.0f;
      }
    }
  }

  cvReleaseImage(&bin_img);
  cvReleaseImage(&big_img);
  cvReleaseImage(&gray_img);
}

// Free the image object and the array of pixels.
ImageData::~ImageData() {
  cvReleaseImage(&img);
  free(pixels);
}

// Return the value of the pixel at (x,y).
float ImageData::pixel(int x, int y) const {
  if (x < 0 || x >= NORMALIZED_WIDTH ||
      y < 0 || y >= NORMALIZED_HEIGHT) {
    return NULL;
  }
  return pixels[y * width + x];
}

int ImageData::getWidth() const {
  return width;
}

int ImageData::getHeight() const {
  return height;
}

void ImageData::getPixels(vector<double> *v) const {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      (*v)[i * width + j] = pixels[i * width + j];
    }
  }
}

bool ImageData::error() const {
  return readError;
}

