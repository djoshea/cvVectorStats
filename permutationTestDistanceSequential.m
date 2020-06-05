function [ pValue, shuffleDistribution ] = permutationTestDistanceSequential( class1, class2, nResamples )
    %class1 and class2 are D { N x T } data, where D is the number of
    % dimensions and N is the number of samples, T is optional time
    %
    %nResamples specifies how many resamplings to perform
    
    nDims = length(class1);
    [allDat, allGroups] = deal(cell(nDims, 1));
    for d = 1:nDims
        allDat{d} = [class1{d}; class2{d}];
        allGroups{d} = [zeros(size(class1{d},1), 1); ones(size(class2{d},1),1)];
    end
    [pValue, shuffleDistribution] = permutationTestSequential(@(dat,groupClasses)(distanceWrapper(dat, groupClasses, false)), allDat, allGroups, nResamples, 'one-sided'); 
end

function dist = distanceWrapper(allDat, allGroups, subtractMean)
    nDims = length(allDat);
    [class1, class2] = deal(cell(nDims, 1));
    for d = 1:nDims
        class1{d} = allDat{d}(allGroups{d}==0,:);
        class2{d} = allDat{d}(allGroups{d}==1,:);
    end
    [euclidianDistance, squaredDistance] = cvDistanceSequential(class1, class2, subtractMean);
    dist = [euclidianDistance, squaredDistance];
end