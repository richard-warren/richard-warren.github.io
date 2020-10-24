This is the code for my personal website, which is forked and modified from the excellent [academic pages template](https://github.com/academicpages/academicpages.github.io).

## notes on mods
- **to change font:** change `$rick-font` in `_sass\_variables.scss`
- **to change accent color:** change `$accent-color` in `_sass\_variables.scss`
- **science and art hacks:** created `archive-portfolio.html`, modified from `archive-single`. it shows images to the left of text, with titles above (see the text beneath the `rick mod` comment in that file).

## other notes
- **image file sizes:** try to keep images beneath 500 kb. Use command line tool [`ImageMagick`](https://imagemagick.org/index.php) to edit image sizes as follows:
  - `convert barfancy.png -resize 300x300 -quality 85% barfancy.jpg`
  - the above line with maintain image aspect ratio by default, making images smaller and compressing as jpg
- **gif file sizes:** file sizes of gifs can be reduced using [`gifsicle`](https://www.lcdf.org/gifsicle/man.html). the following line applies loss compression, and limits the number of colors to 64:
  - `gifsicle -i cellfie_original.gif -O3 --lossy=80 --colors 64 -o cellfie.gif`
