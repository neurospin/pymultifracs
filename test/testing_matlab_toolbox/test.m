clear all
close all

% ========================================================================
% Reads a params.json file and run mf analysis using the Matlab toolbox
% ========================================================================

format long

% -------------------------------------------------------------------------
% Path/file settings
% -------------------------------------------------------------------------

% test folder
[test_dir, ~, ~]  = fileparts(pwd);

% params.json file
params_file = fullfile(test_dir, 'params.json');

% output folder
out_folder = fullfile(test_dir, 'test_outputs');

% test data folder
test_data_dir = fullfile(test_dir, 'test_data');

% files containing test data
test_files = dir(fullfile(test_data_dir, '*.mat'));
test_data_files_short = {};
for i = 1:length(test_files)
    test_data_files_short{i} = test_files(i).name;
end

% output filenames
out_filenames = {};
for i = 1:length(test_data_files_short)
    out_filenames{i} = strcat('matlab_output_',test_data_files_short{i}(1:end-4), '.csv');
end

% -------------------------------------------------------------------------
% Read and decode json file
% -------------------------------------------------------------------------
raw_params = fileread(params_file);
params_struct = jsondecode(raw_params);
clear raw_content
n_params = numel(fieldnames(params_struct));

% -------------------------------------------------------------------------
% Initialize output files
% -------------------------------------------------------------------------
for i = 1:length(test_files)
    out_filename = fullfile(out_folder, out_filenames{i});
    
    cell_header = {};
    len_q       = numel(params_struct.x1.q);
    
    cell_header{1} = 'test_index';
    cell_header{2} = 'hmin';
    cell_header{3} = 'eta_p';
    cell_header{4} = 'c1';
    cell_header{5} = 'c2';
    cell_header{6} = 'c3';
    ind_q = 0;
    for ind_cell = 7:len_q+6
        cell_header{ind_cell} = char(strcat('zeta_', string(ind_q)));
        ind_q = ind_q + 1;
    end
    
    ind_q = 0;
    for ind_cell = len_q+7:2*len_q+6
        cell_header{ind_cell} = char(strcat('hq_', string(ind_q)));
        ind_q = ind_q + 1;
    end
 
    ind_q = 0;
    for ind_cell = 2*len_q+7:3*len_q+6
        cell_header{ind_cell} = char(strcat('Dq_', string(ind_q)));
        ind_q = ind_q + 1;
    end
    
    comma_header = [cell_header;repmat({','},1,numel(cell_header))];
    comma_header = comma_header(:)';
    text_header = cell2mat(comma_header);
    
    %write header to file
    fid = fopen(out_filename,'w');
    fprintf(fid,'%s\n',text_header);
    fclose(fid);
end



for i = 1:length(test_files)
    
    % ---------------------------------------------------------------------
    % Load data
    % ---------------------------------------------------------------------
    filename = fullfile(test_data_dir, test_data_files_short{i});
    file_contents = load(filename);
    data = file_contents.data;
    
    %output file
    out_filename = fullfile(out_folder, out_filenames{i});
    
    
    disp("")
    disp(strcat("* Analyzing file ", filename))
    
    % ---------------------------------------------------------------------
    % Run mf_analysis and save results
    % ---------------------------------------------------------------------
    for test_index = 1:n_params
        
        if mod(test_index,10) == 0
           fprintf("Test %d of %d \n", test_index, n_params)
        end
        
        % remove persistent variables
        clear rlistcoefdaub
        
        % Initialize MF object with global parameters
        mf_obj = MF_BS_tool_inter;
        mf_obj.method_mrq = [1 2];
        mf_obj.cum     = 3;
        mf_obj.verbosity =0;
        
        % set parameters and analyze data
        formalism = 0; % 0 for wavelet leaders, 1 for wavelet coefficients
        
        test_index_str = strcat('x',string(test_index));
        params = params_struct.(test_index_str);
        mf_obj.gamint = params.gamint;
        mf_obj.j1     = params.j1;
        mf_obj.j2     = params.j2;
        mf_obj.nwt    = params.nb_vanishing_moments;
        mf_obj.q      = params.q';
        mf_obj.wtype  = params.wtype;
        mf_obj.p      = params.p;
        if strcmp(mf_obj.p, 'inf')
            mf_obj.p   = Inf;
        elseif strcmp(mf_obj.p, 'none')
            mf_obj.p   = Inf;
            formalism  = 1; % use wavelet coefficients as result
        end
        
        mf_obj.analyze (data);
        
        % Get results
        cid = mf_obj.get_cid ();  % Indices of c_p
        zid = mf_obj.get_zid ();  % Indices of zeta(q)
        hid = mf_obj.get_hid ();  % Indices of h(q)
        Did = mf_obj.get_Did ();  % Indices of D(q)

        if formalism == 0
            cp = mf_obj.est.LWT.t(cid);  % Estimates of c_p
            zq = mf_obj.est.LWT.t(zid);  % Estimates of zeta(q)
            eta_p = mf_obj.est.LWT.zp;    % estimate of eta_p
            % ---- in the python package, DWT is always used to compute
            % hmin
            hmin = mf_obj.est.DWT.h_min;  % estimate of hmin 
            
            hq = mf_obj.est.LWT.t(hid);  % Estimates of h(q)
            Dq = mf_obj.est.LWT.t(Did);  % Estimates of D(q)
        else
            cp = mf_obj.est.DWT.t(cid);  % Estimates of c_p
            zq = mf_obj.est.DWT.t(zid);  % Estimates of zeta(q)
            eta_p = mf_obj.est.DWT.zp;    % estimate of eta_p
            % ---- in the python package, DWT is always used to compute
            % hmin
            hmin = mf_obj.est.DWT.h_min;  % estimate of hmin  
            
            hq = mf_obj.est.DWT.t(hid);  % Estimates of h(q)
            Dq = mf_obj.est.DWT.t(Did);  % Estimates of D(q)
        end
            
        %
        index          = test_index;
        %write data to end of file
        data_to_write = [index, hmin, eta_p, cp(1), cp(2), cp(3), zq(:)', hq(:)', Dq(:)'];
        dlmwrite(out_filename,data_to_write,'-append');
        
        % Clear object
        clear mf_obj
    end
end
