function [ imgMod ] = cropResize( img, xDim, yDim )

dimensions = [xDim,yDim];

            [y,x,planes] = size(img);
            
            % not a square - crop
            if x~=y
                if x<y
                    squareSize = x;
                    bigger = y;
                else
                    squareSize = y;
                    bigger = x;
                end
            
            
            center = [x/2,y/2];
            
            if bigger == x
            topLeftCorner = [center(1)-squareSize/2,1];
            else
            topLeftCorner = [1,center(2)+squareSize/2];
            end
            
            cropArea = [topLeftCorner(1),topLeftCorner(2),squareSize,squareSize];
            
            imgMod = imcrop(img,cropArea);
            imgMod = imresize(imgMod,dimensions);
            else
                %already a square - just resize
                imgMod = imresize(img,dimensions);
            end
            
           

end

