function [refls, trans] = RCWA_Silicon_Gaussian_new(depth, pitch, diameter, acc, show1, n_layers)
% RCWA simulation for Gaussian profile nanoholes in Silicon substrate
% Air region above, Silicon below with Gaussian hole starting at interface
% Parameterized by radius t, with depth z(t) calculated using Gaussian

    % Input validation and parameter setup
    if nargin < 6
        n_layers = 30;
    end
    if nargin < 5
        show1 = 0;
    end
    if nargin < 4
        acc = 11;
    end
    
    % Validate geometry for periodic boundary conditions
    validate_geometry(depth, pitch, diameter);
    
    % Wavelength sampling
    wave = 380:10:780;
    trans = zeros(size(wave));
    refls = zeros(size(wave));
    max_radius = diameter/2;  % Maximum radius at Si surface (interface)
    
    % Add RCWA path
    addpath('RETICOLO V8/reticolo_allege');
    
    % Load Silicon material data
    try
        Si = load('Si.mat');
        WL = Si.WL;
        R = Si.R;
        I = Si.I;
    catch
        error('Cannot load Si.mat file. Check if file exists in current directory.');
    end
    
    % Validate Si.mat wavelength coverage
    if min(WL) > min(wave) || max(WL) < max(wave)
        error('Si.mat wavelengths (%.0f–%.0f nm) do not cover simulation range (%.0f–%.0f nm).', ...
              min(WL), max(WL), min(wave), max(wave));
    end
    
    % Match simulation wavelengths to closest Si.mat wavelengths
    n_Si_values = zeros(size(wave), 'like', 1i);
    for i = 1:length(wave)
        [~, idx] = min(abs(WL - wave(i)));
        n_Si_values(i) = R(idx) + 1i*I(idx);
    end
    
    % Plot raw Si.mat data for validation
    if show1 == 1
        figure('Name', 'Silicon Refractive Index from Si.mat');
        plot(WL, R, 'b-', 'DisplayName', 'Real(n_{Si})');
        hold on;
        plot(WL, I, 'r-', 'DisplayName', 'Imag(n_{Si})');
        scatter(wave, real(n_Si_values), 'b+', 'DisplayName', 'Selected Real(n_{Si})');
        scatter(wave, imag(n_Si_values), 'r+', 'DisplayName', 'Selected Imag(n_{Si})');
        xlabel('Wavelength (nm)');
        ylabel('Refractive Index');
        title('Silicon Refractive Index from Si.mat');
        legend;
        grid on;
        hold off;
    end
    
    % Layer structure definition for Silicon region only
    z_positions = linspace(0, depth, n_layers+1);
    layer_thickness = diff(z_positions);
    z_centers = z_positions(1:end-1) + layer_thickness/2;
    
    % CORRECTED: Gaussian profile that covers full depth range
    % Adjust sigma_t so that the Gaussian covers from near-surface to full depth
    
    layer_radii = zeros(size(z_centers));
    
    % Calculate sigma_t to ensure the Gaussian covers the required depth range
    % We want: at t=max_radius, z should be close to 0 (surface)
    % z(t_max) = depth * exp(-(1)^2 / (2*sigma_t^2)) = target_min_depth
    target_min_depth = min(z_centers) * 0.5;  % Go even shallower than shallowest layer
    sigma_t = sqrt(-1 / (2 * log(target_min_depth / depth)));
    
    % Ensure sigma_t is reasonable (not too small)
    sigma_t = max(sigma_t, 0.3);
    sigma_t = min(sigma_t, 1.2);
    
    % Create a smooth radius sampling from 0 to max_radius
    t_sample = linspace(0, max_radius, 1000);
    
    % Calculate corresponding depths for each radius sample
    z_sample = depth * exp(-(t_sample/max_radius).^2 / (2*sigma_t^2));
    
    % For each layer depth, find the corresponding radius by interpolation
    for i = 1:length(z_centers)
        z_target = z_centers(i);
        
        % Find the radius that gives this depth using interpolation
        if z_target >= max(z_sample)
            % Very shallow depth → small radius (center of hole)
            layer_radii(i) = 0;
        elseif z_target <= min(z_sample)
            % Very deep depth → large radius (edge of hole)
            layer_radii(i) = max_radius;
        else
            % Interpolate to find radius
            % Since z decreases as t increases, we need to flip for interpolation
            z_flipped = flip(z_sample);
            t_flipped = flip(t_sample);
            layer_radii(i) = interp1(z_flipped, t_flipped, z_target, 'linear');
        end
        
        % Ensure reasonable bounds
        layer_radii(i) = max(layer_radii(i), 0.01 * max_radius);
        layer_radii(i) = min(layer_radii(i), max_radius);
    end
    
    layer_diameters = 2 * layer_radii;
    
    % Validate hole sizes
    max_diameter = max(layer_diameters);
    if max_diameter >= pitch
        error('Maximum hole diameter (%.1f nm) exceeds pitch (%.1f nm).', max_diameter, pitch);
    end
    
    % Display simulation information
    fprintf('\n=== Gaussian Nanohole RCWA Simulation (t-parameterized) ===\n');
    fprintf('Structure: Air layer above, Silicon substrate below\n');
    fprintf('Maximum diameter (at surface): %.1f nm (fill factor: %.1f%%)\n', ...
            diameter, 100*(diameter/pitch)^2);
    fprintf('Minimum diameter (at bottom): %.1f nm\n', min(layer_diameters));
    fprintf('Hole depth in Silicon: %.1f nm\n', depth);
    fprintf('Pitch: %.1f nm\n', pitch);
    fprintf('Number of layers: %d\n', n_layers);
    fprintf('Wavelength points: %d\n', length(wave));
    fprintf('Accuracy parameter: %d\n', acc);
    fprintf('Parameterization: z(t) = depth * exp(-(t/r_max)²/(2*σ_t²))\n');
    fprintf('Sigma_t parameter: %.3f (auto-calculated for full depth coverage)\n', sigma_t);
    
    % RETICOLO setup
    [prv,vmax] = retio([],inf*1i);
    
    % ===== PERIODIC BOUNDARY CONDITIONS =====
    periods = [pitch, pitch];
    
    % ===== INCIDENT FIELD SPECIFICATION =====
    n_air = 1;
    angle_theta = 0;
    k_parallel = n_air*sin(angle_theta*pi/180);
    angle_delta = 0;
    
    % ===== RCWA PARAMETERS =====
    parm = res0;
    parm.sym.pol = 1;
    parm.res1.champ = 1;
    
    % ===== SYMMETRY BOUNDARY CONDITIONS =====
    parm.sym.x = 1;
    parm.sym.y = 1;
    
    % ===== TRUNCATION PARAMETERS =====
    nn = [acc, acc];
    
    % ===== LAYER STRUCTURE =====
    % Structure: Air layer + n_layers of Silicon with holes + Silicon substrate
    num_total_layers = 1 + n_layers + 1;
    air_thickness = max(200, 0.5*max(wave));
    substrate_thickness = 90000;
    all_thicknesses = [air_thickness, layer_thickness, substrate_thickness];
    layer_indices = 1:num_total_layers;
    profile = {all_thicknesses, layer_indices};
    
    % Energy conservation tracking
    energy_error = zeros(size(wave));
    failed_wavelengths = [];
    
    % Main wavelength loop
    num_wavelengths = length(wave);
    for i = 1:num_wavelengths
        tic_start = tic;
        wavelength = wave(i);
        n_Si = n_Si_values(i);
        
        % ===== TEXTURE DEFINITION =====
        textures = cell(1, num_total_layers);
        
        % Layer 1: Air (no holes)
        textures{1} = n_air;
        
        % Layers 2 to n_layers+1: Silicon with Gaussian holes filled with air
        for j = 1:n_layers
            current_diameter = layer_diameters(j);
            diameter_ratio = current_diameter / pitch;
            if diameter_ratio > 0.001
                % Silicon background with circular air hole
                textures{j+1} = {n_Si, [0, 0, current_diameter, current_diameter, n_air, 1]};
            else
                % If hole becomes too small, use minimum size for numerical stability
                min_hole_diameter = 0.001 * pitch;
                textures{j+1} = {n_Si, [0, 0, min_hole_diameter, min_hole_diameter, n_air, 1]};
            end
        end
        
        % Last layer: Solid Silicon substrate (no holes)
        textures{n_layers+2} = n_Si;
        
        % ===== RCWA CALCULATION =====
        try
            aa = res1(wavelength, periods, textures, nn, k_parallel, angle_delta, parm);
            two_D = res2(aa, profile);
            trans(i) = sum(two_D.TEinc_top_transmitted.efficiency_TE);
            refls(i) = sum(two_D.TEinc_top_reflected.efficiency_TE);
            energy_sum = trans(i) + refls(i);
            energy_error(i) = abs(energy_sum - 1);
            if energy_error(i) > 0.05
                warning('Energy conservation at λ=%.0f nm: T+R=%.3f (error=%.1f%%)', ...
                        wavelength, energy_sum, 100*energy_error(i));
            end
        catch ME
            warning('RCWA calculation failed at wavelength %.0f nm: %s. Skipping...', ...
                    wavelength, ME.message);
            failed_wavelengths = [failed_wavelengths, i];
            trans(i) = NaN;
            refls(i) = NaN;
            energy_error(i) = NaN;
        end
        
        % Clear memory
        if exist('aa', 'var'), clear aa; end
        if exist('two_D', 'var'), clear two_D; end
        clear textures;
        
        % Progress
        elapsed_time = toc(tic_start);
        if mod(i, 5) == 0 || i == num_wavelengths
            if isnan(trans(i))
                fprintf('λ=%3.0f nm (%2d/%2d): FAILED, Time=%.2fs\n', ...
                        wavelength, i, num_wavelengths, elapsed_time);
            else
                fprintf('λ=%3.0f nm (%2d/%2d): T=%.3f, R=%.3f, Error=%.1f%%, Time=%.2fs\n', ...
                        wavelength, i, num_wavelengths, trans(i), refls(i), ...
                        100*energy_error(i), elapsed_time);
            end
        end
    end
    
    % Handle failed wavelengths
    if ~isempty(failed_wavelengths)
        fprintf('\nWarning: %d wavelengths failed. Setting to zero...\n', length(failed_wavelengths));
        trans(isnan(trans)) = 0;
        refls(isnan(refls)) = 0;
    end
    
    % Calculate absorption
    absorp = 1 - trans - refls;
    
    % Results validation
    fprintf('\n=== Simulation Quality Check ===\n');
    valid_errors = energy_error(~isnan(energy_error));
    if ~isempty(valid_errors)
        fprintf('Average energy conservation error: %.2f%%\n', 100*mean(valid_errors));
        fprintf('Maximum energy conservation error: %.2f%%\n', 100*max(valid_errors));
        if max(valid_errors) > 0.05
            warning('Large energy conservation errors detected.');
        end
    end
    
    % Optional plotting
    if show1 == 1
        wave_full = 380:5:780;
        trans_full = interp1(wave, trans, wave_full, 'pchip');
        refls_full = interp1(wave, refls, wave_full, 'pchip');
        absorp_full = 1 - trans_full - refls_full;
        create_spectral_plot(wave_full, trans_full, refls_full, absorp_full, ...
                           diameter, pitch, depth, n_layers, sigma_t);
        create_radius_parameterized_plot(layer_diameters, z_centers, depth, diameter, sigma_t, max_radius);
    end
    
    % Results summary
    fprintf('\n=== Results Summary ===\n');
    fprintf('Spectral Averages (%.0f-%.0f nm):\n', min(wave), max(wave));
    fprintf('  Average Transmission: %.4f\n', mean(trans));
    fprintf('  Average Reflection: %.4f\n', mean(refls));
    fprintf('  Average Absorption: %.4f\n', mean(absorp));
    
    [max_trans, idx_max_t] = max(trans);
    [max_refl, idx_max_r] = max(refls);
    [max_abs, idx_max_a] = max(absorp);
    
    fprintf('Peak Values:\n');
    fprintf('  Maximum Transmission: %.4f at %.0f nm\n', max_trans, wave(idx_max_t));
    fprintf('  Maximum Reflection: %.4f at %.0f nm\n', max_refl, wave(idx_max_r));
    fprintf('  Maximum Absorption: %.4f at %.0f nm\n', max_abs, wave(idx_max_a));
    
    visible_range = wave >= 400 & wave <= 700;
    if any(visible_range)
        fprintf('Visible Range Averages (400-700 nm):\n');
        fprintf('  Visible Transmission: %.4f\n', mean(trans(visible_range)));
        fprintf('  Visible Reflection: %.4f\n', mean(refls(visible_range)));
        fprintf('  Visible Absorption: %.4f\n', mean(absorp(visible_range)));
    end
    
    % Profile analysis
    fprintf('\nRadius-Parameterized Gaussian Analysis:\n');
    fprintf('  Surface diameter: %.1f nm\n', max(layer_diameters));
    fprintf('  Bottom diameter: %.1f nm\n', min(layer_diameters));
    fprintf('  Sigma_t parameter: %.2f\n', sigma_t);
    fprintf('  Profile equation: z(t) = %.1f * exp(-(t/%.1f)²/(2*%.2f²))\n', depth, max_radius, sigma_t);
    fprintf('  Parameterization: Radius t → Depth z(t)\n');
end

%% Helper Functions

function validate_geometry(depth, pitch, diameter)
    if diameter >= pitch
        error('Hole diameter (%.1f nm) must be smaller than pitch (%.1f nm)', diameter, pitch);
    end
    if diameter <= 0 || pitch <= 0 || depth <= 0
        error('All dimensions must be positive');
    end
    fill_factor = (diameter/pitch)^2;
    if fill_factor > 0.64
        warning('High fill factor (%.1f%%) may cause hole interactions', 100*fill_factor);
    end
    aspect_ratio = depth/diameter;
    if aspect_ratio > 10
        warning('High aspect ratio (%.1f) may require finer discretization', aspect_ratio);
    end
    if depth > 5*pitch
        warning('Very deep structure (%.1f × pitch) may need more layers', depth/pitch);
    end
end

function create_spectral_plot(wave, trans, refls, absorp, diameter, pitch, depth, n_layers, sigma_t)
    figure('Name', 'RCWA Spectral Response', 'Position', [100, 100, 900, 600]);
    plot(wave, trans, '-', 'Color', [0, 0.447, 0.741], 'LineWidth', 2, 'DisplayName', 'Transmission');
    hold on
    plot(wave, refls, '-', 'Color', [0.85, 0.325, 0.098], 'LineWidth', 2, 'DisplayName', 'Reflection');
    plot(wave, absorp, '-', 'Color', [0.929, 0.694, 0.125], 'LineWidth', 2, 'DisplayName', 'Absorption');
    xlabel('Wavelength (nm)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Spectral Response', 'FontSize', 14, 'FontWeight', 'bold');
    title('Radius-Parameterized Gaussian Nanohole Array in Silicon', 'FontSize', 16, 'FontWeight', 'bold');
    annotation_text = {
        sprintf('D_{max} = %.0f nm', diameter);
        sprintf('P = %.0f nm', pitch);
        sprintf('H = %.0f nm', depth);
        sprintf('Layers = %d', n_layers);
        sprintf('σ_t = %.2f', sigma_t);
        'z(t) parameterization'
    };
    text(0.75, 0.85, annotation_text, 'Units', 'normalized', 'FontSize', 12, ...
         'BackgroundColor', 'white', 'EdgeColor', 'black', 'LineWidth', 1);
    xlim([min(wave), max(wave)]);
    ylim([0, 1]);
    legend('Location', 'best', 'FontSize', 12);
    grid on
    grid minor
    set(gca, 'FontSize', 12, 'LineWidth', 1);
    hold off
end

function create_radius_parameterized_plot(layer_diameters, z_centers, depth, diameter, sigma_t, max_radius)
    figure('Name', 'Radius-Parameterized Gaussian Profile', 'Position', [150, 150, 1200, 500]);
    
    % Subplot 1: Cross-sectional view
    subplot(1, 3, 1);
    
    % Create smooth profile for visualization
    t_smooth = linspace(0, max_radius, 200);
    z_smooth = depth * exp(-(t_smooth/max_radius).^2 / (2*sigma_t^2));
    % Note: No clamping needed since sigma_t is auto-calculated for full range
    
    % Plot the hole profile (both sides)
    fill([t_smooth, -flip(t_smooth)], [z_smooth, flip(z_smooth)], ...
         [0.8, 0.9, 1], 'EdgeColor', 'blue', 'LineWidth', 2, 'DisplayName', 'Air hole');
    hold on;
    
    % Plot layer boundaries
    for i = 1:length(layer_diameters)
        radius = layer_diameters(i) / 2;
        z = z_centers(i);
        plot([-radius, radius], [z, z], 'r--', 'LineWidth', 1);
        plot([radius, radius], [z-depth/40, z+depth/40], 'ro', 'MarkerSize', 4);
        plot([-radius, -radius], [z-depth/40, z+depth/40], 'ro', 'MarkerSize', 4);
    end
    
    % Add silicon background
    silicon_width = max_radius * 1.5;
    fill([-silicon_width, -max_radius, max_radius, silicon_width, silicon_width, -silicon_width], ...
         [0, 0, 0, 0, depth, depth], [0.7, 0.7, 0.7], 'EdgeColor', 'black', ...
         'LineWidth', 1, 'DisplayName', 'Silicon');
    
    xlabel('Radius (nm)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Depth (nm)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Cross-sectional View', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    axis equal;
    set(gca, 'YDir', 'reverse'); % Depth increases downward
    xlim([-silicon_width*1.1, silicon_width*1.1]);
    ylim([0, depth]);
    
    % Subplot 2: Radius vs depth profile  
    subplot(1, 3, 2);
    plot(2*t_smooth, z_smooth, 'b-', 'LineWidth', 3, 'DisplayName', 'z(t) Gaussian');
    hold on;
    scatter(layer_diameters, z_centers, 50, 'ro', 'filled', 'DisplayName', 'Layer centers');
    
    xlabel('Hole Diameter (nm)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Depth (nm)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Radius-Parameterized Profile', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    set(gca, 'YDir', 'reverse'); % Depth increases downward
    xlim([0, diameter*1.1]);
    ylim([0, depth]);
    
    % Subplot 3: Parameter relationship
    subplot(1, 3, 3);
    t_norm = linspace(0, 1, 200);
    z_norm = exp(-(t_norm).^2 / (2*sigma_t^2));
    % No clamping needed with auto-calculated sigma_t
    
    plot(t_norm, z_norm, 'g-', 'LineWidth', 3, 'DisplayName', 'Normalized z(t)');
    hold on;
    
    % Show effect of different sigma_t values
    sigma_values = [0.3, 0.5, 0.7];
    colors = {'r--', 'b--', 'm--'};
    for i = 1:length(sigma_values)
        sig = sigma_values(i);
        z_temp = exp(-(t_norm).^2 / (2*sig^2));
        % No clamping needed
        plot(t_norm, z_temp, colors{i}, 'LineWidth', 2, 'DisplayName', sprintf('σ_t = %.1f', sig));
    end
    
    xlabel('Normalized Radius t/r_{max}', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Normalized Depth z/z_{max}', 'FontSize', 12, 'FontWeight', 'bold');
    title('Parameter Relationship', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    xlim([0, 1]);
    ylim([0, 1]);
    
    % Add equation text
    text(0.5, 0.8, sprintf('z(t) = z_{max} × exp(-(t/r_{max})²/(2σ_t²))'), ...
         'Units', 'normalized', 'FontSize', 10, 'FontWeight', 'bold', ...
         'BackgroundColor', 'yellow', 'EdgeColor', 'black', 'LineWidth', 1, ...
         'HorizontalAlignment', 'center');
    
    text(0.5, 0.65, {sprintf('Current: σ_t = %.3f', sigma_t), 'Auto-calculated for full coverage'}, ...
         'Units', 'normalized', 'FontSize', 9, ...
         'BackgroundColor', [0.8 1.0 0.8], 'EdgeColor', 'black', 'LineWidth', 1, ...
         'HorizontalAlignment', 'center');
    
    % Adjust subplot spacing
    sgtitle('Radius-Parameterized Gaussian Nanohole Analysis', 'FontSize', 16, 'FontWeight', 'bold');
    
    hold off;
end