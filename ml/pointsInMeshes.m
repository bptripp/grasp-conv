% Creates random points within meshes and saves them to a CSV file. The 
% meshes are translated to put their bounding box centres at 0,0,0. 
% This code requires grasp_db code to be in the Matlab path. 

objDir = '/Users/bptripp/code/grasp-conv/data/obj_tmp';

javaaddpath '/Users/bptripp/code/grasp_db/src/matlab/matlab/ray/ray.jar'

fid = fopen('/Users/bptripp/code/grasp-conv/data/obj-points.csv','w');

files = dir(objDir);
for i = 1:length(files)
    if length(files(i).name) > 4 && strcmp(files(i).name(end-3:end), '.obj')        
        obj = read_wobj(sprintf('%s/%s', objDir, files(i).name));
        [tree, v] = getAABBTree(obj);
        bb = tree.myBoundingBox;
        centre = mean(bb,1)';
        
        nPoints = 5;
        for j = 1:nPoints
            p = getRandomPointInVolume(tree) - centre;
            fprintf(fid, '%s, %f, %f, %f\n', files(i).name, p);
        end
    end
end

fclose(fid);
