function testSpreadProjectedSequential()
    %%
    %The following samples data from eight multivariate distributions whose mean vectors
    %lie in a 2D ring in D dimensions. Then, the euclidean distance of each mean vector
    %from the centroid of all mean vectors is estimated with either the
    %cross-validated spread metric or the standard way. 
    %
    % for simultaneous data, we have: vSpread( obs, classIdx, ...)
    % where obs is an N x D matrix, where N is the number of observations and D is
    %the number of dimensions. classIdx is an N x 1 vector describing, for
    %each observation, the class to which it belongs. 
    % 
    % for sequential data we instead have cvSpreadSequential( data, classIdx, ... )
    % where data  D x 1 { N_d x T }, classIdx D x 1 {N_d x 1} 
    % where D is the number of dimensions and N_d is the number of samples for dimension d for all classes
    % classIdx{d} class index for  each trial in data{d} 
    
    % true latents lie on 2d ring 
    nDims = 100;
    nTime = 90; % must be at least 40
    nConditions = 8;
    
    nDimsProj = 10;
    W = randn(nDims, nDimsProj);
%     W = eye(nDims);
    
    nReps = 100;
    rng(1);

    % time course of tuning strength
    xAxis = linspace(-2,2, nTime-40)';
    tuningProfile = normpdf(xAxis,0,1);
    tuningProfile = tuningProfile - min(tuningProfile);
    tuningProfile = [zeros(20,1); tuningProfile; zeros(20,1)];
    
    % tuning by condition - C x 1
    theta = linspace(0,2*pi, nConditions + 1)';
    theta = theta(1:nConditions);

    spreadEst = zeros(nTime, nReps);
    spreadEstUnbiased = zeros(nTime, nReps);
    tuningVec = randn(nDims,2); % projection mat to high d space

    % max_rates_dc is (Dx2 * 2xC --> DxC , which is then multiplied by tuningProfile(t)
    max_rates_dc = (tuningVec*[cos(theta'); sin(theta')]);
    
    for r=1:nReps
        [data, conditions] = deal(cell(nDims, 1));
        means = nan(nConditions, nTime, nDims);
        
        for d = 1:nDims
            this_signal_ct = (max_rates_dc(d, :) .* tuningProfile)'; % 1xC * Tx1 --> TxC --> CxT
        
            nTrialsByC = randi(20, nConditions, 1) + 15;
            totalTrials = sum(nTrialsByC);
            
            conditions_this = nan(totalTrials, 1);
            rates_this = nan(totalTrials, nTime);
            trialOffset = 0;
            for c = 1:nConditions
                idx_c = trialOffset + (1:nTrialsByC(c));
                trialOffset = trialOffset + nTrialsByC(c);
                
                conditions_this(idx_c) = c;
                rates_this(idx_c, :) = repmat(this_signal_ct(c, :), nTrialsByC(c), 1) + randn(nTrialsByC(c), nTime);
                means(c, :, d) = mean(rates_this(idx_c, :), 1);
            end
            
            data{d} = rates_this;
            conditions{d} = conditions_this;
        end
        
        centroid = mean(means, 1); % C x T x D --> 1 x T x D
        
        % take 2-norm over dimensions, mean over conditions --> 1 x T --> T x 1
        deltas = TensorUtils.linearCombinationAlongDimension(means - centroid, 3, W'); % C x T x N --> C x T x K
        spreadEst(:,r) = mean(vecnorm(deltas, 2, 3), 1)';
        spreadEstUnbiased(:,r) = cvSpreadProjectedSequential( data, conditions, W' );
    end
    
    true_rates = permute(max_rates_dc, [2 3 1]) .* shiftdim(tuningProfile, -1); % C x T x D
    true_centroid = mean(true_rates, 1);
    true_delta = TensorUtils.linearCombinationAlongDimension(true_rates - true_centroid, 3, W');
    trueSpread = mean(vecnorm(true_delta, 2, 3), 1)';
        
    %%
    %plot results
    tvec = linspace(0,1,nTime);

    colors = [0.8 0 0;
        0 0 0.8];
    lHandles = zeros(2,1);

    figure('Position',[680   838   659   260]);
    hold on;

    [mn,sd,CI] = normfit(spreadEst');
    [mn_un,sd_un,CI_un] = normfit(spreadEstUnbiased');

    lHandles(1)=plot(tvec, mn, 'Color', colors(1,:), 'LineWidth', 2);
    lHandles(2)=plot(tvec, mn_un, 'Color', colors(2,:), 'LineWidth', 2);
    lHandles(3)=plot(tvec,trueSpread,'--k','LineWidth',2);

    plot(tvec', [mn'-sd', mn'+sd'], 'Color', colors(1,:), 'LineStyle', '--');
    plot(tvec', [mn_un'-sd_un', mn_un'+sd_un'], 'Color', colors(2,:), 'LineStyle', '--');

    title(['Projected sequential spread']);
    xlabel('Time');
    ylabel('Spread');

    legend(lHandles, {'Standard','Cross-Validated','True Spread'},'Box','Off');
%     saveas(gcf,[plotDir 'spreadTrueVsEstimated.png'],'png');
    
    %%
    %The following samples data from eight multivariate distributions whose mean vectors
    %lie in a 2D ring. Then, the euclidean distance of each mean vector
    %from the centroid of all mean vectors is estimated, along with the
    %confidence interval (using 3 methods). The coverage of the confidence
    %intervals is verified. 

    return;
    if testCI
        nReps = 100;
        nDims = 100;
        tuningStrength = linspace(0,0.5,3);
        isCovered_cen = zeros(length(tuningStrength), nReps);
        isCovered_per = zeros(length(tuningStrength), nReps);
        isCovered_jac = zeros(length(tuningStrength), nReps);
        trueSpread = zeros(length(tuningStrength),1);

        for tuningIdx=1:length(tuningStrength)
            disp(tuningIdx);

            mnVectors = zeros(8,nDims);
            for conIdx=1:8
                mnVectors(conIdx,:) = tuningVec*[cos(theta(conIdx)); sin(theta(conIdx))]*tuningStrength(tuningIdx);
            end

            centroid = mean(mnVectors);
            distFromCentroid = zeros(8,1);
            for conIdx=1:8
                distFromCentroid(conIdx) = norm(mnVectors(conIdx,:)-centroid);
            end

            trueSpread(tuningIdx) = mean(distFromCentroid);

            for n=1:nReps
                allData = [];
                allGroups = [];
                for conIdx=1:8
                    if conIdx<=4
                        nTrials = 13;
                    elseif conIdx==5
                        nTrials = 31;
                    else
                        nTrials = 20;
                    end
                    signal = (tuningVec*[cos(theta(conIdx)); sin(theta(conIdx))])'*tuningStrength(tuningIdx);
                    allData = [allData; repmat(signal, nTrials, 1) + randn(nTrials,nDims)];
                    allGroups = [allGroups; repmat(conIdx, nTrials, 1)];
                end

                allMN = zeros(8, nDims);
                for conIdx=1:8
                    allMN(conIdx,:) = mean(allData(allGroups==conIdx,:));
                end
                centroid = mean(allMN);

                distFromCentroid = zeros(8,1);
                for conIdx=1:8
                    distFromCentroid(conIdx) = norm(allMN(conIdx,:)-centroid);
                end

                [stat, ~, CI] = cvSpread( allData, allGroups, 'bootCentered', 0.05, 1000 );
                isCovered_cen(tuningIdx,n) = trueSpread(tuningIdx)>CI(1,1) & trueSpread(tuningIdx)<CI(2,1);

                [stat, ~, CI] = cvSpread( allData, allGroups, 'bootPercentile', 0.05, 1000 );
                isCovered_per(tuningIdx,n) = trueSpread(tuningIdx)>CI(1,1) & trueSpread(tuningIdx)<CI(2,1);

                [stat, ~, CI] = cvSpread( allData, allGroups, 'jackknife', 0.05, 1000 );
                isCovered_jac(tuningIdx,n) = trueSpread(tuningIdx)>CI(1,1) & trueSpread(tuningIdx)<CI(2,1);
            end
        end

        coverageCell = {isCovered_per, isCovered_cen, isCovered_jac};
        ciNames = {'Percentile Bootstrap','Centered Bootstrap','Jackknife'};
        plotCICoverage( ciNames, coverageCell, trueSpread );
        saveas(gcf,[plotDir 'spreadCICoverage.png'],'png');
    end
    
    %%
    %The following samples data from eight multivariate distributions whose mean vectors
    %lie in a 2D ring. Then, the euclidean distance of each mean vector
    %from the centroid of all mean vectors is estimated, along with the
    %p-value using a permutation test. The # of significant (p<0.05) runs
    %is checked. 
    
    if testPermTest
        nReps = 100;
        nDims = 100;
        tuningStrength = linspace(0,0.5,3);
        pValues = zeros(length(tuningStrength), nReps);
        trueSpread = zeros(length(tuningStrength),1);

        for tuningIdx=1:length(tuningStrength)
            disp(tuningIdx);

            mnVectors = zeros(8,nDims);
            for conIdx=1:8
                mnVectors(conIdx,:) = tuningVec*[cos(theta(conIdx)); sin(theta(conIdx))]*tuningStrength(tuningIdx);
            end

            centroid = mean(mnVectors);
            distFromCentroid = zeros(8,1);
            for conIdx=1:8
                distFromCentroid(conIdx) = norm(mnVectors(conIdx,:)-centroid);
            end

            trueSpread(tuningIdx) = mean(distFromCentroid);

            for n=1:nReps
                allData = [];
                allGroups = [];
                for conIdx=1:8
                    if conIdx<=4
                        nTrials = 13;
                    elseif conIdx==5
                        nTrials = 31;
                    else
                        nTrials = 20;
                    end
                    signal = (tuningVec*[cos(theta(conIdx)); sin(theta(conIdx))])'*tuningStrength(tuningIdx);
                    allData = [allData; repmat(signal, nTrials, 1) + randn(nTrials,nDims)];
                    allGroups = [allGroups; repmat(conIdx, nTrials, 1)];
                end

                allMN = zeros(8, nDims);
                for conIdx=1:8
                    allMN(conIdx,:) = mean(allData(allGroups==conIdx,:));
                end
                centroid = mean(allMN);

                distFromCentroid = zeros(8,1);
                for conIdx=1:8
                    distFromCentroid(conIdx) = norm(allMN(conIdx,:)-centroid);
                end

                pValues(tuningIdx, n) = permutationTestSpread( allData, allGroups, 1000 );
            end
        end

        sigCell = {pValues<0.05};
        sigNames = {'Permutation Test'};
        plotPermutationTestResults( sigNames, sigCell, trueSpread );
        saveas(gcf,[plotDir 'spreadPermutationTest.png'],'png');
    end
end

