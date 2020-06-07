function [bigFoldMatrices, smallFold, nFolds] = getSequentialFoldIndicatorMatrices( obsCounts, varargin )
    % An internal function used to split data into folds for cross-validation for use with sequentially recorded data. 
    % obsCounts is nClasses x nDims matrix of observation counts.
    % foldMatrices is  nClasses x nDims { nFolds x obsByDimByClass(c, d) } logical indicating whether to include
    % a given observation from that class, dimension in a 
    % for computing the means
    %
    % bigFoldMatrices is C x D { nFolds x obsCounts(c,d) }
    % if smallAsMatrices == false (default), smallFold is C x D { nFolds numeric or cell of indices, numeric with strategy lcm, cell with strategy min } 
    % else if smallAsMatrices == true, smallFold is also C x D { nFolds x obsCounts(c,d) }
    
    
    p = inputParser();
    p.addParameter('normalizeForAveraging', false, @islogical);
    p.addParameter('maxFolds', NaN, @isscalar);
    p.addParameter('strategy', 'lcm', @(x) ischar(x) || isstring(x));
    p.addParameter('smallAsMatrices', false, @islogical); % true: return smallFold same format as bigFoldMatrices, false return vector of 
    p.parse(varargin{:});
    normalizeForAveraging = p.Results.normalizeForAveraging;
    maxFolds = p.Results.maxFolds;
    if isnan(maxFolds)
        maxFolds = 2 * max(obsCounts, [], 'all');
    end
    strategy = string(p.Results.strategy);
    smallAsMatrices = p.Results.smallAsMatrices;
    
    minObs = min(obsCounts, [], 'all');
    assert(minObs >= 2, 'Minimum of 2 observations per dimension required for corss-validation');
    
    [nClasses, nDims] = size(obsCounts);
    
    switch strategy
        case 'lcm'
            nFolds = lcm_multi(obsCounts(:), maxFolds);

            nFolds = min(nFolds, maxFolds);

            [bigFoldMatrices, smallFold] = deal(cell(nClasses, nDims));
            for c=1:nClasses
                for d = 1:nDims
                    n = obsCounts(c, d);
                    if smallAsMatrices
                        if normalizeForAveraging
                            smallFold{c,d} = build_striped_oneHot(nFolds, n, 1/(n-1));
                        else
                            smallFold{c,d} = build_striped_oneHot(nFolds, n, 1);
                        end
                    else
                        smallFold{c, d} = mod(uint32(0:nFolds-1)', n) + 1;
                    end
                    if normalizeForAveraging
                        bigFoldMatrices{c,d}    = build_striped_allExceptOneHot(nFolds, n, 1/(n-1));
                    else
                        bigFoldMatrices{c,d}    = build_striped_allExceptOneHot(nFolds, n, 1);
                    end
                end
            end
        case 'min'
            % same as non-sequential jackknife
            nFolds = min(obsCounts, [], 'all');
            [bigFoldMatrices, smallFold] = deal(cell(nClasses, nDims));
            
            for c=1:nClasses
                for d = 1:nDims
                    smallFoldIdx = getFoldedIdx(obsCounts(c, d), nFolds);
                    n = obsCounts(c, d);
                    bigFoldMatrices{c,d} = build_idx_each_row(nFolds, n, smallFoldIdx, normalizeForAveraging, true);
                    if smallAsMatrices
                        smallFold{c,d} = build_idx_each_row(nFolds, n, smallFoldIdx, normalizeForAveraging, false);
                    else
                        smallFold{c,d} = smallFoldIdx';
                    end
                end
            end
        otherwise
            error('Unknown fold strategy %s', strategy);
    end
end

function y = lcm_multi(vals, maxVal)
    y = vals(1);
    for i = 2:numel(vals)
        y = lcm(y, vals(i));
        if y > maxVal
            y = maxVal;
            return;
        end
    end
end

function mat = build_striped_oneHot(rows, cols)
    % build a rows x cols matrix consisting of stacked identity matrices with 1s set to value
    
    inds = repmat((1:cols:rows)', 1, cols) + (0:cols-1)*(rows+1);
    [i, j] = ind2sub([rows cols], inds(inds<rows*cols));
    mat = sparse(i, j, true, rows, cols);
    
end

function mat = build_striped_allExceptOneHot(rows, cols, value)
    % build a rows x cols matrix consisting of stacked identity matrices with 1s set to value
    
    inds = repmat((1:cols:rows)', 1, cols) + (0:cols-1)*(rows+1);
    mat = repmat(value, rows, cols);
    mat(inds(inds <= rows*cols)) = 0;
end

function mat = build_idx_each_row(rows, cols, idx_each_row, row_normalized, idx_complement)
    if row_normalized
        mat = zeros(rows, cols);
        if idx_complement
            for r = 1:rows
                idx_complement = setdiff(1:cols, idx_each_row{r});
                mat(r, idx_complement) = 1 / numel(idx_complement);
            end
        else
            for r = 1:rows
                mat(r, idx_each_row{r}) = 1 ./ numel(idx_each_row{r});
            end
        end
    else
        mat = false(rows, cols);
        if idx_complement
            for r = 1:rows
                idx_complement = setdiff(1:cols, idx_each_row{r});
                mat(r, idx_complement) = true;
            end
        else
            for r = 1:rows
                mat(r, idx_each_row{r}) = false;
            end
        end
    end   
end
        
    
    
        