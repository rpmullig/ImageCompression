// Package for loading photos for compression purposes

package png

import (
	"image"
	"image/png"
	"os"
)

// The Image represents a structure for working with PNG images.
type Image struct {
	In  image.Image
}

// Load returns a Image that was loaded based on the filePath parameter
func Load(filePath string) (*Image, error) {

	inReader, err := os.Open(filePath)

	if err != nil {
		return nil, err
	}
	defer inReader.Close()

	inImg, err := png.Decode(inReader)
	
	if err != nil {
		return nil, err
	}

	return &Image{inImg}, nil
}

func Save(img image.Image, filePath string) error {

	outWriter, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer outWriter.Close()

	err = png.Encode(outWriter, img)
	if err != nil {
		return err
	}
	return nil
}