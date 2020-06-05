function [ pValue, nullDistribution ] = permutationTestSequential( fun, allDat, allGroups, nResamples, mode  )
    %An internal function for permutation testing. Returns the p-value and
    %null distribution.
    
    %fun is the test statistic function with two inputs: a matrix of
    %observations and a vector where each entry describes which
    %distribution that observation belongs to
    
    %allDat is an D { N x T } cell of observations, where N is the number of
    % combined observations and D is the number of dimensions (T is optional timepoints)
    
    %allGroups is an D { N } cell vector describing which class ("group") each
    %observation belongs to
    
    %nResamples specifies how many resamplings to perform
    
    %mode can be one-sided or two-sided
    
    if nargin<5
        mode='one-sided';
    end
    
    nDims = length(allDat);
    unshuffledStatistic = fun(allDat, allGroups);
    nullDistribution = zeros(nResamples, length(unshuffledStatistic));
    
    for n=1:nResamples
        shuffleGroups = allGroups;
        for d = 1:nDims
            shuffIdx = randperm(length(allGroups{d}));
            shuffleGroups{d} = allGroups{d}(shuffIdx);
        end
        nullDistribution(n,:) = fun(allDat, shuffleGroups);
    end
           
    if strcmp(mode,'one-sided')
        [~,sortIdx] = sort(nullDistribution(:,1));
        nullDistribution = nullDistribution(sortIdx,:);
    
        pValue = zeros(length(unshuffledStatistic),1);
        for n=1:length(unshuffledStatistic)
            pIdx = find(nullDistribution(:,n)>unshuffledStatistic(n),1,'first');
            if isempty(pIdx)
                pValue(n) = 1/nResamples;
            else
                pValue(n) = (nResamples-pIdx)/nResamples;
            end
        end
    elseif strcmp(mode,'two-sided')
        [~,sortIdx] = sort(abs(nullDistribution(:,1)));
        nullDistribution = nullDistribution(sortIdx,:);
    
        pValue = zeros(length(unshuffledStatistic),1);
        for n=1:length(unshuffledStatistic)
            pIdx = find(abs(nullDistribution(:,n))>abs(unshuffledStatistic(n)),1,'first');
            if isempty(pIdx)
                pValue(n) = 1/nResamples;
            else
                pValue(n) = (nResamples-pIdx)/nResamples;
            end
        end        
    else
        error('Wrong mode, should be one-sided or two-sided');
    end
end

