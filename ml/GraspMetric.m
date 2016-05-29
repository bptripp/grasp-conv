classdef GraspMetric < handle
    % Calculates a fast grasp metric based on ray tracing. 
    
    properties (Access = public)
        tree;
        collisionPoints;
        rayPoints;
        rayDir;
    end

    methods (Access = public)
        function gm = GraspMetric(objFile)
            javaaddpath '/Users/bptripp/code/grasp_db/src/matlab/matlab/ray/ray.jar'            
            
            gm.tree = GraspMetric.getAABBTree(objFile);
            gm.collisionPoints = GraspMetric.getCollisionPoints();
            gm.rayPoints = GraspMetric.getRayPoints();
            gm.rayDir = [0; -1; 0];
        end
        
        function plotMetricPoints(gm, gripperPosition, gripperRotationMatrix)        
            c = movePoints(gm.collisionPoints, gripperRotationMatrix, gripperPosition);
            r = movePoints(gm.rayPoints, gripperRotationMatrix, gripperPosition);

            internalPoints = zeros(3,1000);
            for i = 1:size(internalPoints,2)
                internalPoints(:,i) = getRandomPointInVolume(gm.tree);
            end
            scatter3(internalPoints(1,:), internalPoints(2,:), internalPoints(3,:), 'k.')
            hold on
            
            scatter3(c(1,:), c(2,:), c(3,:))
            scatter3(r(1,:), r(2,:), r(3,:), 'r')
            axis equal
        end
                
        function score = getScore(gm, gripperPosition, gripperRotationMatrix)
            cp = movePoints(gm.collisionPoints, gripperRotationMatrix, gripperPosition);
            rp = movePoints(gm.rayPoints, gripperRotationMatrix, gripperPosition);
            rd = gripperRotationMatrix * gm.rayDir;
            
            t = gm.tree;
            
            collision = 0;
            for i = 1:size(cp, 2)
                if ca.uwaterloo.nrlab.ray.AABBTree.isInVolume(t, cp(:,i)) 
                    collision = 1;
                    break;
                end
            end
            
            score = 0;
            if ~collision  
                for i = 1:size(rp,2)                    
                    if t.intersects(rp(:,i), rd) 
                        score = score + 1;
                    end
                end
            end
            score = score / size(rp,2);
        end
       
    end
    
    methods (Static)
        
        function tree = getAABBTree(objFile)
            obj = read_wobj(objFile);
            [tree, ~] = getAABBTree(obj);
        end
        
        function points = getRayPoints()
            % Points from which to draw rays to estimate whether various 
            % parts of fingers will close on object.

            np = 15;
            x = [-.04 0 .04];
            y = 0.1;
            z = linspace(0, .13, np)';  
            points = [x(1)*ones(np,1) y*ones(np,1) z; x(2)*ones(np,1) y*ones(np,1) z; x(3)*ones(np,1) y*ones(np,1) z]';
        end
            
        function points = getCollisionPoints()
            % Points in partially closed gripper at which to test collision 
            % with object ... 
            
            np = 5;
            [X, Y] = meshgrid(linspace(-.055, .055, np), linspace(-.045, .045, np));  
            palmPoints = [reshape(X, numel(X), 1) reshape(Y, numel(Y), 1) zeros(numel(Y), 1)];

            nf = 10;
            distanceAlongFinger = linspace(0, .13, nf);
            fingerYZ = distanceAlongFinger'*[cos(pi/3) sin(pi/3)] + [.0425*ones(nf,1) zeros(nf,1)];    
            fingerPoints = [-.04*ones(nf,1) fingerYZ; .04*ones(nf,1) fingerYZ; zeros(nf,1) fingerYZ.*repmat([-1 1], nf, 1)];

            points = [palmPoints; fingerPoints]';
        end
        
        function points = getRandomPoints(n, radius, surface)
            pointDirections = randn(3, n);
            norms = sum(pointDirections.^2, 1).^.5;
            points = radius * pointDirections ./ repmat(norms, 3, 1);

            if ~surface
                minStandoff = .1; % fraction of radius
                points = points .* repmat(minStandoff + (1-minStandoff)*rand(1,n), 3, 1); % uniform distance from centre (not uniform 3D density)
            end
        end
        
        function R = getRotationMatrix(point, roll)
            % Rotation matrix to orient to 0,0,0 from point, with roll
            % about axis pointing to 0,0,0.
            
            z = -point; %location of (0,0,0) relative to point

            alpha = atan(z(2)/z(1));
            if z(1) < 0, alpha = alpha + pi; end
            if alpha < 0, alpha = alpha + 2*pi; end

            % rotate by alpha about z
            Rz = [[cos(alpha), -sin(alpha), 0]; [sin(alpha), cos(alpha), 0]; [0, 0, 1]];

            % find elevation in new coordinates
            beta = -atan(sqrt(z(1)^2+z(2)^2)/z(3));
            if z(3) < 0, beta = beta + pi; end
            if beta < 0, beta = beta + 2*pi; end
            %TODO indices

            % rotate by beta about y
            Ry = [[cos(beta), 0, -sin(beta)]; [0, 1, 0]; [sin(beta), 0, cos(beta)]];

            gamma = roll;
            Rz2 = [[cos(-gamma), -sin(-gamma), 0]; [sin(-gamma), cos(-gamma), 0]; [0, 0, 1]];

            R = Rz * Ry * Rz2;
        end

    end
    
end

function result = movePoints(points, rot, offset)
    result = rot * points + repmat(offset, 1, size(points,2));
end
