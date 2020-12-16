function [ euclideanDistance, squaredDistance, CI, CIDistribution ] = cvSpreadSequential( data, classIdx, subtractMean, CIMode, CIAlpha, CIResamples )
    %This function estimates the distance between the means of two
    %distributions.

    %inputs:
    % data  D x 1 { N_d x T } , 
    % where D is the number of dimensions and N_d is the number of samples for dimension d for all classes

    % classIdx D x 1 {N_d x 1} 
    % classIdx{d} class index for  each trial in data{d} 

    %If subtractMean is true, this will center each vector
    %before computing the size of the difference (default is off).
    
    %CIMode can be none, bootCentered, bootPercentile, or jackknife
    
    %CIAlpha sets the coverage of the confidence interval to
    %100*(1-CIAlpha) percent
    
    %CIResamples sets the number of bootstrap resamples, if using bootstrap
    %mode (as opposed to jackknife)
    
    %CIDistribution is the distribution of bootstrap statistics or
    %jackknife leave-one-out statistics
    
    %outputs:
    
    %The first column of CI corresponds to euclidean distance, the second
    %column corresponds to squared distance. 
    
    %CIDistribution is the bootstrap distribution or leave-one-out
    %jackknife estimates

    if nargin<3
        subtractMean = false;
    end
    
    assert(~subtractMean, 'subtractMean not yet implemented');

    if nargin<4
        CIMode = 'none';
    end
    if nargin<5
        CIAlpha = 0.05;
    end
    if nargin<6
        CIResamples = 1000;
    end
    
    assert(iscell(data) && iscell(classIdx));
    
    classList = unique(cat(1, classIdx{:}));
    nClasses = numel(classList);
    nDims = numel(data);
    nTime = size(data{1}, 2);
    
    % first we split the data by class
    dataByClass = cell(nClasses, nDims);
    obsMat = nan(nClasses, nDims);
    for d = 1:nDims
        for c = 1:nClasses
            mask = classIdx{d}==classList(c);
            obsMat(c, d) = nnz(mask);
            dataByClass{c, d} = data{d}(mask, :);
        end
    end
    
    [bigFoldMatrices, smallFoldMatrices, nFolds, nObsBig, nObsSmall] = getSequentialFoldIndicatorMatrices(obsMat); % C x D { nFolds x obsMat(c, d) } logical indicator matrices
    
    % if we multiply foldMatrices{1, d} * class1{d}, we have sum of observations included in each fold as nFolds x T
    % if instead of using foldMatrices directly, we normalize by the number of included rows to average appropriately
    % want to assmelbe these set-wise means into F x T x D matrices, which can then be reshaped and multiplied
    
    squaredDistByDim = nan(nDims, nTime);
   
    % for each fold,  we want to compute the delta from each 
    for d = 1:nDims
        [bigMeans, smallMeans] = deal(nan(nFolds(d), nTime, nClasses));
        for c = 1:nClasses
            bigMeans(:, :, c) = bigFoldMatrices{c, d} * dataByClass{c,d} ./ nObsBig{c,d}; % (F x nObs) * (nObs x T) --> F x T
            smallMeans(:, :, c) = smallFoldMatrices{c, d} * dataByClass{c,d} ./ nObsSmall{c,d}; % (F x nObs) * (nObs x T) --> F x T
        end
        bigCentroid = mean(bigMeans, 3); % F x T x C --> F x T x 1
        smallCentroid = mean(smallMeans, 3); % F x T x C --> F x T x 1
        
        squaredDistByDim(d, :) = sum( (bigMeans - bigCentroid) .* (smallMeans - smallCentroid), [1 3]) ./ (nClasses * nFolds(d));  % F x T x C --> 1 x T x 1
    end
    
    squaredDistance = sum(squaredDistByDim, 1); % D x T --> 1 x T
    euclideanDistance = sign(squaredDistance).*sqrt(abs(squaredDistance)); % 1 x T
    
    %compute confidence interval if requensted
    if ~strcmp(CIMode, 'none')
        % the confidence interval functions cvJackknifeCI and cvBootstrapCI expect a single input that is 
        
        % dataByClass is nClasses x nDims { nObs x T }
        % cvCISequential needs nClasses { nDims { nObs x T } } 
        classCell = cell(nClasses,1);
        for n=1:nClasses
            classCell{n} = dataByClass(c, :)';
        end
        
        [CI, CIDistribution] = cvCISequential([euclideanDistance; squaredDistance], @(varargin) ciWrapper(subtractMean, varargin{:}), classCell, CIMode, CIAlpha, CIResamples);
    else
        CI = [];
        CIDistribution = [];
    end
end

function output = ciWrapper(subtractMean, varargin)
     % varargin is nClasses { nDims { nObs x T } }
     % cvSpreadSequential needs nDims { nObs x T } for all classes combined
     dataFlat = cat(2, varargin{:}); % D x C
     nDims = size(dataFlat, 1);
     data = cell(nDims, 1);
     classIdx = cell(nDims, 1);
     for d = 1:nDims
         [data{d}, classIdx{d}] = catWhich(1, dataFlat{d, :});
     end
     
    [ euclideanDistance, squaredDistance ] = cvSpreadSequential( data, classIdx, subtractMean );
    output = [euclideanDistance; squaredDistance];
end

function [out, which] = catWhich(dim, varargin)
    % works like cat, but returns a vector indicating which of the
    % inputs each element of out came from
    out = cat(dim, varargin{:});
    if nargout > 1
        which = cell2mat(makecol(cellfun(@(in, idx) idx*ones(size(in, dim), 1), varargin, ...
            num2cell(1:numel(varargin)), 'UniformOutput', false)));
    end
end

function vec = makecol( vec )

    % transpose if it's currently a row vector (unless its 0 x 1, keep as is)
    if (size(vec,2) > size(vec, 1) && isvector(vec)) && ~(size(vec, 1) == 0 && size(vec, 2) == 1)
        vec = vec';
    end

    if size(vec, 1) == 1 && size(vec, 2) == 0
        vec = vec';
    end

end



