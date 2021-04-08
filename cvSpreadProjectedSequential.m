function [ euclideanDistance, squaredDistance, CI, CIDistribution ] = cvSpreadProjectedSequential( data, classIdx, projectionKbyD, subtractMean, CIMode, CIAlpha, CIResamples)
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

    if nargin<4
        subtractMean = false;
    end
    if nargin<5
        CIMode = 'none';
    end
    if nargin<6
        CIAlpha = 0.05;
    end
    if nargin<7
        CIResamples = 10000;
    end
    assert(iscell(data) && iscell(classIdx));
    
    classList = unique(cat(1, classIdx{:}));
    nClasses = numel(classList);
    nDims = numel(data);
    nTime = size(data{1}, 2);
    W = projectionKbyD';
    
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
    
    obsMat = cellfun(@(x) size(x, 1), dataByClass); % C x D
    [bigFoldMatrices, smallFoldMatrices, nFoldsByDim, nObsBig, nObsSmall] = getSequentialFoldIndicatorMatrices(obsMat); % C x D { nFolds x obsMat(c, d) } logical indicator matrices
    
    % compute the big (Delta) and small (delta) fold differences for each fold (for each dimension)
    % if we multiply foldMatrices{1, d} * class1{d}, we have sum of observations included in each fold as nFolds x T
    % if instead of using foldMatrices directly, we normalize by the number of included rows to average appropriately
    % want to assmelbe these set-wise means into F x T x D matrices, which can then be reshaped and multiplied
    [mean_Delta, mean_delta] = deal(nan(nDims, nTime, nClasses)); 
    A = nan(nDims, nDims, nTime); % D x D x T
    for d = 1:nDims
        [bigMeans, smallMeans] = deal(nan(nFoldsByDim(d), nTime, nClasses));
        for c = 1:nClasses
            bigMeans(:, :, c) = bigFoldMatrices{c, d} * dataByClass{c,d} ./ nObsBig{c,d}; % (F x nObs) * (nObs x T) --> F x T
            smallMeans(:, :, c) = smallFoldMatrices{c, d} * dataByClass{c,d} ./ nObsSmall{c,d}; % (F x nObs) * (nObs x T) --> F x T
        end
        bigCentroid = mean(bigMeans, 3); % F x T x C --> F x T x 1
        smallCentroid = mean(smallMeans, 3); % F x T x C --> F x T x 1
        
        % fill in the diagonal term of A
        A(d, d, :) = shiftdim( sum( (bigMeans - bigCentroid) .* (smallMeans - smallCentroid), [1, 3]) ./ (nClasses * nFoldsByDim(d)), -1); % F x T x C --> 1 x T --> 1 x 1 xT
        
        % needed for the off-diagonal d ~= d' terms below
        mean_Delta(d, :, :) = mean(bigMeans - bigCentroid, 1);     % F x T x C --> 1 x T x C
        mean_delta(d, :, :) = mean(smallMeans - smallCentroid, 1); % F x T x C --> 1 x T x C
    end

    % now loop over each pair of dimensions to fill in the off-diagonal terms
    for d = 1:nDims
        for e = 1:nDims
            if d == e, continue, end
            A(d, e, :) = shiftdim(mean(mean_Delta(d, :, :) .* mean_delta(e, :, :), 3), -1); % 1 x T x C --> 1 x T --> 1 x 1 x T
        end
    end

    % Now we have A (D x D x T) and want to compute D^2
    squaredDistance = nan(1, nTime);
    for t = 1:nTime
        squaredDistance(t) = trace(W' * A(:, :, t) * W);
    end
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

