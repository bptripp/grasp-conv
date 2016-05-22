% Creates random points within meshes and saves them to a CSV file. The 
% meshes are translated to put their bounding box centres at 0,0,0. 
% This code requires grasp_db code to be in the Matlab path. 

objDir = '/Users/bptripp/code/grasp-conv/data/obj_files';
nPoints = 10;

javaaddpath '/Users/bptripp/code/grasp_db/src/matlab/matlab/ray/ray.jar'

fid = fopen('/Users/bptripp/code/grasp-conv/data/obj-points.csv','w');

doVisualCheck = 0;

files = dir(objDir);
for i = 1:length(files)
    if length(files(i).name) > 4 && strcmp(files(i).name(end-3:end), '.obj')        
        fprintf('Processing %s\n', files(i).name)
        obj = read_wobj(sprintf('%s/%s', objDir, files(i).name));
        [tree, v] = getAABBTree(obj);
        bb = tree.myBoundingBox;
        centre = mean(bb,1)';
        
        if doVisualCheck 
            points = zeros(3,1000);
            for j = 1:1000
                points(:,j) = getRandomPointInVolume(tree) - centre;
            end
            figure, hold on
            scatter3(points(1,:), points(2,:), points(3,:))
        end

        for j = 1:nPoints
            p = getRandomPointInVolume(tree) - centre;
            fprintf(fid, '%s, %i, %f, %f, %f\n', files(i).name, (j-1), p);
            
            if doVisualCheck, scatter3(p(1), p(2), p(3), 'ro'), end
        end                
        
    end
end

fclose(fid);
