function create_nanohole_dataset(num_samples, filename)
% CREATE_NANOHOLE_DATASET_CONTINUOUS - Generate diverse dataset with continuous sampling
% 
% Inputs:
%   num_samples   - Number of data points to generate (default: 5000)
%   filename      - Output filename (default: 'nanohole_dataset_continuous.mat')
%   use_continuous- Use continuous sampling (default: true)
%
% KEY IMPROVEMENTS:
%   - CONTINUOUS parameter sampling (no step sizes!)
%   - Much larger variance and diversity
%   - Better coverage of parameter space
%   - Stratified sampling for uniform distribution

    if nargin < 1
        num_samples = 5000;  % Increased default
    end
    if nargin < 2
        filename = 'nanohole_dataset_continuous.mat';
    end
    % Always use continuous sampling (this is the enhanced version)
    
    fprintf('=== Enhanced Silicon Nanohole Array Dataset Generation ===\n');
    fprintf('Target samples: %d\n', num_samples);
    fprintf('Sampling mode: CONTINUOUS (enhanced version)\n');
    fprintf('Output file: %s\n', filename);
    
    %% ENHANCED Parameter Ranges (CONTINUOUS SAMPLING)
    % Expanded ranges for better diversity
    diameter_min =400;    % nm (expanded from 350)
    diameter_max =500;   % nm (expanded from 550)
    
    pitch_min = 501;      % nm (expanded from 650) 
    pitch_max = 600;      % nm (expanded from 900)
    
    depth_min = 600;       % nm (same)
    depth_max = 1000;      % nm (expanded from 300)
    
    fprintf('\nEXPANDED Parameter Ranges:\n');
    fprintf('  Diameter: %d - %d nm (CONTINUOUS)\n', diameter_min, diameter_max);
    fprintf('  Pitch:    %d - %d nm (CONTINUOUS)\n', pitch_min, pitch_max);
    fprintf('  Depth:    %d - %d nm (CONTINUOUS)\n', depth_min, depth_max);
    
    %% Generate Diverse Parameter Combinations
    fprintf('\nGenerating diverse parameter combinations...\n');
    
    % CONTINUOUS uniform sampling (always)
    fprintf('Using CONTINUOUS uniform sampling...\n');
    
    % Generate many more candidates than needed for filtering
    candidate_multiplier = 1;  % Generate 3x more than needed
    num_candidates = num_samples * candidate_multiplier;
    
    % Uniform random sampling in expanded ranges
    candidate_diameters = diameter_min + (diameter_max - diameter_min) * rand(num_candidates, 1);
    candidate_pitches = pitch_min + (pitch_max - pitch_min) * rand(num_candidates, 1);
    candidate_depths = depth_min + (depth_max - depth_min) * rand(num_candidates, 1);
    
    % Filter for valid geometries
    valid_mask = false(num_candidates, 1);
    
    fprintf('Filtering for valid geometries...\n');
    for i = 1:num_candidates
        if is_valid_geometry_enhanced(candidate_diameters(i), ...
                                    candidate_pitches(i), ...
                                    candidate_depths(i))
            valid_mask(i) = true;
        end
        
        if mod(i, 10000) == 0
            fprintf('  Processed %d/%d candidates, found %d valid\n', ...
                    i, num_candidates, sum(valid_mask(1:i)));
        end
    end
    
    % Extract valid combinations
    valid_diameters = candidate_diameters(valid_mask);
    valid_pitches = candidate_pitches(valid_mask);
    valid_depths = candidate_depths(valid_mask);
    
    fprintf('Found %d valid combinations from %d candidates\n', ...
            length(valid_diameters), num_candidates);
    
    % Sample the requested number
    if length(valid_diameters) >= num_samples
        % Use stratified sampling for better space coverage
        sample_indices = stratified_sampling_continuous(...
            valid_diameters, valid_pitches, valid_depths, num_samples);
        
        diameters = valid_diameters(sample_indices);
        pitches = valid_pitches(sample_indices);
        depths = valid_depths(sample_indices);
    else
        fprintf('Warning: Only %d valid combinations found, using all\n', ...
                length(valid_diameters));
        diameters = valid_diameters;
        pitches = valid_pitches;
        depths = valid_depths;
        num_samples = length(diameters);
    end
    
    %% Calculate Enhanced Derived Parameters
    fill_factors = (diameters ./ pitches).^2;
    aspect_ratios = depths ./ diameters;
    
    % Additional derived parameters for better feature diversity
    pitch_to_diameter_ratio = pitches ./ diameters;
    depth_to_pitch_ratio = depths ./ pitches;
    volume_fraction = fill_factors .* (depths ./ pitches);  % 3D fill factor
    
    %% Print Enhanced Statistics
    fprintf('\n=== ENHANCED Dataset Statistics ===\n');
    fprintf('Samples: %d\n', length(diameters));
    
    fprintf('GEOMETRIC PARAMETERS:\n');
    fprintf('  Diameter:     %.1f ± %.1f nm (range: %.1f - %.1f)\n', ...
            mean(diameters), std(diameters), min(diameters), max(diameters));
    fprintf('  Pitch:        %.1f ± %.1f nm (range: %.1f - %.1f)\n', ...
            mean(pitches), std(pitches), min(pitches), max(pitches));
    fprintf('  Depth:        %.1f ± %.1f nm (range: %.1f - %.1f)\n', ...
            mean(depths), std(depths), min(depths), max(depths));
    
    fprintf('DERIVED PARAMETERS:\n');
    fprintf('  Fill Factor:      %.3f ± %.3f (range: %.3f - %.3f)\n', ...
            mean(fill_factors), std(fill_factors), min(fill_factors), max(fill_factors));
    fprintf('  Aspect Ratio:     %.2f ± %.2f (range: %.2f - %.2f)\n', ...
            mean(aspect_ratios), std(aspect_ratios), min(aspect_ratios), max(aspect_ratios));
    fprintf('  Pitch/Diameter:   %.2f ± %.2f (range: %.2f - %.2f)\n', ...
            mean(pitch_to_diameter_ratio), std(pitch_to_diameter_ratio), ...
            min(pitch_to_diameter_ratio), max(pitch_to_diameter_ratio));
    fprintf('  Volume Fraction:  %.4f ± %.4f (range: %.4f - %.4f)\n', ...
            mean(volume_fraction), std(volume_fraction), ...
            min(volume_fraction), max(volume_fraction));
    
    % Compare variance improvement
    fprintf('\n🎉 VARIANCE IMPROVEMENT vs DISCRETE:\n');
    fprintf('  Diameter variance: %.1f (vs ~900 for discrete steps)\n', var(diameters));
    fprintf('  Pitch variance:    %.1f (vs ~6000 for discrete steps)\n', var(pitches));
    fprintf('  Depth variance:    %.1f (vs ~6000 for discrete steps)\n', var(depths));
    
    %% Initialize Output Arrays (same as original)
    wavelengths = 380:10:780;
    num_wavelengths = length(wavelengths);
    
    reflectance_spectra = zeros(length(diameters), num_wavelengths);
    transmittance_spectra = zeros(length(diameters), num_wavelengths);
    computation_times = zeros(length(diameters), 1);
    simulation_status = cell(length(diameters), 1);
    
    %% Run RCWA Simulations (same as original)
    fprintf('\n=== Starting RCWA Simulations ===\n');
    fprintf('Wavelength range: %d - %d nm (%d points)\n', ...
            min(wavelengths), max(wavelengths), num_wavelengths);
    
    accuracy = 7;
    show_plots = 0;
    n_layers = 12;
    
    tic_total = tic;
    failed_simulations = 0;
    
    for i = 1:length(diameters)
        tic_sim = tic;
        
        try
            [refls, trans] = RCWA_Silicon_Gaussian_new(depths(i), pitches(i), ...
                                                  diameters(i), accuracy, ...
                                                  show_plots, n_layers);
            
            reflectance_spectra(i, :) = refls;
            transmittance_spectra(i, :) = trans;
            simulation_status{i} = 'Success';
            
        catch ME
            failed_simulations = failed_simulations + 1;
            reflectance_spectra(i, :) = NaN;
            transmittance_spectra(i, :) = NaN;
            simulation_status{i} = sprintf('Failed: %s', ME.message);
            
            fprintf('  WARNING: Simulation %d failed: %s\n', i, ME.message);
        end
        
        computation_times(i) = toc(tic_sim);
        
        % Progress reporting
        if mod(i, 100) == 0 || i == length(diameters)
            elapsed = toc(tic_total);
            eta = elapsed * (length(diameters) - i) / i;
            success_rate = 100 * (i - failed_simulations) / i;
            
            fprintf('Progress: %d/%d (%.1f%%) | Success: %.1f%% | ETA: %.1fmin\n', ...
                    i, length(diameters), 100*i/length(diameters), success_rate, eta/60);
        end
    end
    
    %% Save Enhanced Dataset
    fprintf('\nSaving enhanced dataset...\n');
    
    dataset = struct();
    
    % Input parameters (CONTINUOUS values!)
    dataset.inputs.diameter = diameters;
    dataset.inputs.pitch = pitches;
    dataset.inputs.depth = depths;
    
    % Enhanced derived parameters
    dataset.derived.fill_factor = fill_factors;
    dataset.derived.aspect_ratio = aspect_ratios;
    dataset.derived.pitch_to_diameter_ratio = pitch_to_diameter_ratio;
    dataset.derived.depth_to_pitch_ratio = depth_to_pitch_ratio;
    dataset.derived.volume_fraction = volume_fraction;
    
    % Outputs (same as before)
    dataset.outputs.wavelengths = wavelengths;
    dataset.outputs.reflectance = reflectance_spectra;
    dataset.outputs.transmittance = transmittance_spectra;
    
    % Enhanced metadata
    dataset.metadata.num_samples = length(diameters);
    dataset.metadata.sampling_method = 'continuous';
    dataset.metadata.successful_simulations = length(diameters) - failed_simulations;
    dataset.metadata.failed_simulations = failed_simulations;
    dataset.metadata.computation_times = computation_times;
    dataset.metadata.simulation_status = simulation_status;
    dataset.metadata.total_computation_time = toc(tic_total);
    dataset.metadata.generation_date = datestr(now);
    
    % Parameter ranges (expanded)
    dataset.parameter_ranges.diameter = [diameter_min, diameter_max];
    dataset.parameter_ranges.pitch = [pitch_min, pitch_max];
    dataset.parameter_ranges.depth = [depth_min, depth_max];
    dataset.parameter_ranges.sampling_type = 'continuous_uniform';
    
    save(filename, 'dataset', '-v7.3');
    
    fprintf('Enhanced dataset saved successfully!\n');
    fprintf('\n🚀 KEY IMPROVEMENTS:\n');
    fprintf('   ✅ CONTINUOUS parameter sampling (no steps!)\n');
    fprintf('   ✅ Much higher parameter variance\n');
    fprintf('   ✅ Expanded parameter ranges\n');
    fprintf('   ✅ Better space coverage\n');
    fprintf('   ✅ Additional derived parameters\n');
    
    % Create enhanced summary plots
    create_enhanced_summary_plots(dataset);
end

%% Enhanced Helper Functions

function valid = is_valid_geometry_enhanced(diameter, pitch, depth)
    % Enhanced geometry validation with relaxed constraints
    
    valid = true;
    
    % 1. Diameter must be smaller than pitch
    if diameter >= pitch
        valid = false;
        return;
    end
    
    % 2. Relaxed fill factor constraint (was 0.5, now 0.6)
    %fill_factor = (diameter/pitch)^2;
    %if fill_factor > 0.6  % More permissive
     %   valid = false;
     %   return;
    %end
    
    % 3. Relaxed aspect ratio constraints
    %aspect_ratio = depth/diameter;
    %if aspect_ratio > 15 || aspect_ratio < 0.1  % More permissive range
    %    valid = false;
    %    return;
    %end
    
    % 4. Minimum meaningful hole depth
    if depth < 20  % Minimum 20nm depth
        valid = false;
        return;
    end
end

function sample_indices = stratified_sampling_continuous(diameters, pitches, depths, num_samples)
    % Enhanced stratified sampling for continuous parameters
    
    if length(diameters) <= num_samples
        sample_indices = 1:length(diameters);
        return;
    end
    
    % Create 3D grid for stratification
    num_bins = ceil(num_samples^(1/3)) + 2;  % Slightly more bins
    
    % Create bins based on quantiles for better distribution
    d_edges = quantile(diameters, linspace(0, 1, num_bins+1));
    p_edges = quantile(pitches, linspace(0, 1, num_bins+1));
    h_edges = quantile(depths, linspace(0, 1, num_bins+1));
    
    % Assign to bins
    d_bins = discretize(diameters, d_edges);
    p_bins = discretize(pitches, p_edges);
    h_bins = discretize(depths, h_edges);
    
    % Remove any NaN assignments (edge cases)
    valid_assignments = ~isnan(d_bins) & ~isnan(p_bins) & ~isnan(h_bins);
    
    d_bins = d_bins(valid_assignments);
    p_bins = p_bins(valid_assignments);
    h_bins = h_bins(valid_assignments);
    valid_indices = find(valid_assignments);
    
    % Create composite bin indices
    bin_indices = sub2ind([num_bins, num_bins, num_bins], d_bins, p_bins, h_bins);
    
    % Sample from each bin
    unique_bins = unique(bin_indices);
    samples_per_bin = max(1, floor(num_samples / length(unique_bins)));
    
    sample_indices = [];
    
    for bin_id = unique_bins'
        bin_mask = (bin_indices == bin_id);
        bin_members = valid_indices(bin_mask);
        
        n_from_bin = min(samples_per_bin, length(bin_members));
        
        if n_from_bin == length(bin_members)
            selected = bin_members;
        else
            perm = randperm(length(bin_members), n_from_bin);
            selected = bin_members(perm);
        end
        
        sample_indices = [sample_indices; selected];
    end
    
    % Adjust to exact number needed
    if length(sample_indices) > num_samples
        perm = randperm(length(sample_indices), num_samples);
        sample_indices = sample_indices(perm);
    elseif length(sample_indices) < num_samples
        % Add random additional samples
        remaining = setdiff(valid_indices, sample_indices);
        additional_needed = num_samples - length(sample_indices);
        if length(remaining) >= additional_needed
            perm = randperm(length(remaining), additional_needed);
            sample_indices = [sample_indices; remaining(perm)];
        end
    end
    
    sample_indices = sort(sample_indices);
end

function create_enhanced_summary_plots(dataset)
    % Enhanced visualization with variance comparisons
    
    try
        figure('Name', 'Enhanced Dataset Analysis', 'Position', [100, 100, 1400, 1000]);
        
        diameters = dataset.inputs.diameter;
        pitches = dataset.inputs.pitch;
        depths = dataset.inputs.depth;
        fill_factors = dataset.derived.fill_factor;
        
        % Parameter distributions with variance annotations
        subplot(2,3,1);
        histogram(diameters, 50, 'FaceColor', [0.2, 0.6, 0.8], 'FaceAlpha', 0.7);
        title(sprintf('Diameter (var=%.0f)', var(diameters)));
        xlabel('Diameter (nm)'); ylabel('Count'); grid on;
        
        subplot(2,3,2);
        histogram(pitches, 50, 'FaceColor', [0.8, 0.4, 0.2], 'FaceAlpha', 0.7);
        title(sprintf('Pitch (var=%.0f)', var(pitches)));
        xlabel('Pitch (nm)'); ylabel('Count'); grid on;
        
        subplot(2,3,3);
        histogram(depths, 50, 'FaceColor', [0.4, 0.8, 0.3], 'FaceAlpha', 0.7);
        title(sprintf('Depth (var=%.0f)', var(depths)));
        xlabel('Depth (nm)'); ylabel('Count'); grid on;
        
        subplot(2,3,4);
        scatter(diameters, pitches, 20, depths, 'filled', 'MarkerFaceAlpha', 0.6);
        xlabel('Diameter (nm)'); ylabel('Pitch (nm)');
        title('Parameter Space Coverage'); colorbar;
        grid on; axis equal;
        
        subplot(2,3,5);
        scatter3(diameters, pitches, depths, 20, fill_factors, 'filled');
        xlabel('Diameter'); ylabel('Pitch'); zlabel('Depth');
        title('3D Parameter Space'); colorbar;
        grid on;
        
        subplot(2,3,6);
        histogram(fill_factors, 50, 'FaceColor', [0.8, 0.6, 0.2], 'FaceAlpha', 0.7);
        title(sprintf('Fill Factor (var=%.4f)', var(fill_factors)));
        xlabel('Fill Factor'); ylabel('Count'); grid on;
        
        fprintf('Enhanced summary plots created.\n');
        
    catch ME
        fprintf('Could not create enhanced plots: %s\n', ME.message);
    end
end