function [estimates]=resample2d_T_S(coef, param_bs, param_est, Circular);
% function [estimates]=resample2d_T_S(leader, param_bs, param_est, Circular);
%   Space - Scale Bootstrap 2d
% 
% Herwig Wendt, Lyon, 2006 - 2008


% try 
%     J1flag; 
% catch; 
%     J1flag=1; 
% end;
J1flag = param_bs.flag_bs_range;
nj = coef.nj;

try 
    Circular; 
catch; 
    Circular=1; 
end;

%-InputError1='param_bs must be a structure with elements:\n   param_bs.n_resamp_1 \n   param_bs.n_resamp_2\n   param_bs.block_size\n   param_bs.ci_method\n   param_bs.verb (optional)\n';
%-try  NB1=isreal(param_bs.n_resamp_1);      if ~NB1; param_bs.n_resamp_1=99; end;  catch error(InputError1, NB1); end
%-try  NB2=isreal(param_bs.n_resamp_2);      if ~NB2; param_bs.n_resamp_2=1;  end; catch error(InputError1, NB2); end
%-try  NB3=isreal(param_bs.block_size);   if ~NB3; param_bs.blocklength=floor(nj(1)/32); end;  catch error(InputError1, NB3); end
%-try  NB4=isreal(param_bs.ci_method);        if ~NB4; param_bs.ci_method=[1:6];end;  catch error(InputError1, NB4); end
%-try CHR=ischar(param_est.fhandle); catch error('The structure param_est must contain a field  param_est.fhandle with a valid function handle'); end
%-if ~CHR; error('The function handle param_est.fhandle is not valid'); end

FHandle=param_est.fhandle;   % Estimator name
%FHandle=str2func(fhandle);       % Function handle for Estimator
NOUT=nargout(FHandle);           % Number of Output Arguments of Estimator
% Check if Estimator needs extra parameters
% fparam=1;
% if length(fieldnames(param_est))==1
%     fparam=0;
% elseif isempty(param_est.param)
%     fparam=0;
% end

EstFun=param_est.estimate_select;
estimate_select=bin2dec(num2str(EstFun));

if (estimate_select == 1)||(estimate_select == 4); NOUT=1;
elseif (estimate_select == 2)||(estimate_select == 5); NOUT=2;
elseif (estimate_select == 3)||(estimate_select == 6); NOUT=3;
else; NOUT=4; end

% Check which Bootstrap statistics are to be calculated
ci_method=param_bs.ci_method; CIdisp=[]; Hstat=[];
if find(ci_method==1); NOR=1; else NOR=0; end
if find(ci_method==2); BAS=1; else BAS=0; end
if find(ci_method==3); PER=1; else PER=0; end
if find(ci_method==4); STU=1; else STU=0; end
if find(ci_method==5); BASADJ=1; else BASADJ=0; end
if find(ci_method==6); PERADJ=1; else PERADJ=0; end

if STU|PERADJ|BASADJ; doB2=1; else; doB2=0; end

B1=param_bs.n_resamp_1;        % number of primary bootstrap resamples
B2=param_bs.n_resamp_2;        % number of bootstrap resamples for variance estimates (used for Normal, Studentised, Variance Stabilising Transformation)
Block=param_bs.block_size;  % Block length for moving blocks resampling

if length(Block)>1; BlockX=Block(1); BlockY=Block(2); else; BlockX=Block; BlockY=Block; end

% Determine is input is wavelet coefficients or leaders
if ndims (coef.value{1}) == 3
    use_dwt = 1;
    [Nx, Ny, ~] = size (coef.value{1});
else
    use_dwt = 0;
    [Nx, Ny] = size (coef.value{1});
end

% try leader.value;
%     COEF=0;     % Leaders
%     [Nx,Ny]=size(leader(1).value.x);
% catch
%     COEF=1;     % Coefficients - don't have field "value"
%     [Nx,Ny]=size(leader(1).x);
% end

Jmax=length(coef.value);    % number of scales

% Initialize coefficients for blocking at each scale:
%   step 1: D[J] = NaN( size(D[1]) );
%   step 2: D[J](1:2^(J-1):end) = Tx[J];
% for j=1:Jmax
%     if COEF
%         DAT(j).X=NaN(Nx,Ny); DAT(j).Y=NaN(Nx,Ny); DAT(j).XY=NaN(Nx,Ny);
%         [tempNx, tempNy]=size(leader(j).x);
%         FACT=2^(j-1);
%         DAT(j).X(1:FACT:tempNx*FACT,1:FACT:tempNy*FACT)=leader(j).x;
%         DAT(j).Y(1:FACT:tempNx*FACT,1:FACT:tempNy*FACT)=leader(j).y;
%         DAT(j).XY(1:FACT:tempNx*FACT,1:FACT:tempNy*FACT)=leader(j).xy;
%     else
%         DAT(j).X=NaN(Nx,Ny);
%         [tempNx, tempNy]=size(leader(j).value.x);
%         FACT=2^(j-1);
%         DAT(j).X(1:FACT:tempNx*FACT,1:FACT:tempNy*FACT)=max(leader(j).value.x, max(leader(j).value.y, leader(j).value.xy));
%     end
%     [DAT(j).N(1), DAT(j).N(2)]=size(DAT(j).X);
% end
% INITIALIZE COEFFICIENTS - NEW (border effects !)
% does take into account shift due to border effects
% does NOT YET take into account shift due to asymmetric wavelet
% HW, Lyon, 08/10/2007
for j=1:Jmax
    if use_dwt
        DAT(j).X=NaN(Nx,Ny); DAT(j).Y=NaN(Nx,Ny); DAT(j).XY=NaN(Nx,Ny);
        [tempNx, tempNy, ~]=size(coef.value{j});
        FACT=2^(j-1);
        tempL=tempNx*FACT-(FACT-1);
        tempshift=ceil((Nx-tempL)/2);
        DAT(j).X(1+tempshift:FACT:tempNx*FACT+tempshift,1+tempshift:FACT:tempNy*FACT+tempshift)=coef.value{j}(:,:,1);
        DAT(j).Y(1+tempshift:FACT:tempNx*FACT+tempshift,1+tempshift:FACT:tempNy*FACT+tempshift)=coef.value{j}(:,:,2);
        DAT(j).XY(1+tempshift:FACT:tempNx*FACT+tempshift,1+tempshift:FACT:tempNy*FACT+tempshift)=coef.value{j}(:,:,3);
    else
        DAT(j).X=NaN(Nx,Ny);
        [tempNx, tempNy]=size(coef.value{j});
        FACT=2^(j-1);
        tempL=tempNx*FACT-(FACT-1);
        tempshift=ceil((Nx-tempL)/2);
        %DAT(j).X(1+tempshift:FACT:tempNx*FACT+tempshift,1+tempshift:FACT:tempNy*FACT+tempshift)=max(leader(j).value.x, max(leader(j).value.y, leader(j).value.xy));
        DAT(j).X(1+tempshift:FACT:tempNx*FACT+tempshift,1+tempshift:FACT:tempNy*FACT+tempshift)=coef.value{j};
    end
    [DAT(j).N(1), DAT(j).N(2)]=size(DAT(j).X);
end

BLOCK=max(Block); % Blocks are square

%% INITIALIZE BLOCKS OF ROW AND COLUMN INDICES
if BLOCK>1
    try % fast but memory intensive
        % initialize blocks of indices:
        %   BX - Blocks of row indices
        %   BX - Blocks of column indices
        if Circular; bx=0:Nx-1; else; bx=0:Nx-BlockX; end
        BX=repmat(bx,BlockX,1);
        addit=repmat([0:BlockX-1]', 1,bx(end)+1);
        BX=BX+addit;
        if Circular; BX=mod(BX,bx(end)+1)+1; else; BX=BX+1; end
        if (Ny~=Nx)|(BlockX~=BlockY)
            if Circular; by=0:Ny-1; else; by=0:Ny-BlockY; end
            BY=repmat(by,BlockY,1);
            addit=repmat([0:BlockY-1]', 1,by(end)+1);
            BY=BY+addit;
            if Circular; BY=mod(BY,by(end)+1)+1; else; BY=BY+1; end
        else
            BY=BX;
        end
    catch % slow but memory save
        % ADAPT OLD CODE HERE
        error('Doing Blocks --> Out of memory: Adapt old method (HW 25/12/06, St. Ulrich)');
    end

    % bootstrap index range
    if ~Circular;
        N_rangex=max(Nx-BlockX+1,1);
        N_rangey=max(Ny-BlockY+1,1);
    else
        N_rangex=Nx;
        N_rangey=Ny;
    end
    % number of blocks to draw
    N_BSx=ceil(Nx/BlockX);
    N_BSy=ceil(Ny/BlockY);
    % number of resamples
    N_resamplex=N_BSx*BlockX;
    N_resampley=N_BSy*BlockY;
end

%% draw indices for resamples
if BLOCK==1
    index = fix(rand(B1,Nx*Ny)*Nx*Ny)+1 ;
    if doB2;
        index2 = fix(rand(B2,B1,Nx*Ny)*Nx*Ny)+1 ;
    end
else
    for bsid=1:B1
        % index for position in image
        tempid = fix(rand(1,N_BSx*N_BSy)*N_rangex*N_rangey)+1;
        tempidxx = rem(tempid-1,Nx)+1   ;  % row block index
        tempidxy = ceil(tempid/Nx) ;       % column block index
        tempBX=repmat(BX(:,tempidxx),BlockY,1);
        tempidx=reshape(tempBX,1,[]);
        tempBY=repmat(reshape(BY(:,tempidxy),[],1),1,BlockX);
        tempidy=reshape(tempBY',[],1)';
        % !!!! shouldn't Nx be N_rangex here (when not circular) (HW, 30/01/2007, Lyon)
        index(bsid,:)=tempidx+(tempidy-1)*Nx; % calculate position of samples for this resample
        if doB2;
            for bs2id=1:B2
                tempid2  = tempid(fix(rand(1,length(tempid))*length(tempid))+1);
                tempidxx2 = rem(tempid2-1,Nx)+1   ;  % row block index
                tempidxy2 = ceil(tempid2/Nx) ;       % column block index
                tempBX2=repmat(BX(:,tempidxx2),BlockY,1);
                tempidx2=reshape(tempBX2,1,[]);
                tempBY2=repmat(reshape(BY(:,tempidxy2),[],1),1,BlockX);
                tempidy2=reshape(tempBY2',[],1)';
                % !!!! shouldn't Nx be N_rangex here (when not circular) (HW, 30/01/2007, Lyon)
                index2(bs2id,bsid,:)=tempidx2+(tempidy2-1)*Nx; % calculate position of samples for this resample
            end
        end
    end
end

EQUAL_LENGTH=1;

for j=1:Jmax
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculate Structure Functions
%     t=cell(1,NOUT);
%     if COEF;
%         tmpcoef=leader(j).vector.all;
%     else
%         tmpcoef=leader(j).vector.max;
%     end
% %     if ~fparam
% %         %[t_out{:}]=feval(fhandle,X);
% %         [t{:}]=FHandle(tmpcoef);
% %     else
%         %[t{:}]=FHandle(tmpcoef,param_est.param);
%         [t{:}]=FHandle(tmpcoef,param_est);
% %     end
%     estimates{j}.t=cell2mat(t);
    % get length of estimates
%     for ii=1:NOUT
%         LEst{j}(ii)=length(t{ii});
%     end
    
    if j>=J1flag
        T=cell(B1,NOUT);
        if doB2; 
            TT=cell(B2,B1,NOUT);
        end
        
        if use_dwt
            xtmp  = coef.value{j}(:,:,1);
            ytmp  = coef.value{j}(:,:,2);
            xytmp = coef.value{j}(:,:,3);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for bsid=1:B1
            % get the coefficients at the indices for each resample
            if use_dwt
                tempDAT=[(DAT(j).X(index(bsid,:))), (DAT(j).Y(index(bsid,:))), (DAT(j).XY(index(bsid,:))), ];
            else
                tempDAT=(DAT(j).X(index(bsid,:)));
            end
            % sort out NaNs
            tempDAT=tempDAT(~isnan(tempDAT));

            % Catch case of empty resample (mainly last scale): use ordinary
            % bootstrap sample
            if isempty(tempDAT);
                if use_dwt
                    Ntemp=prod(size(xtmp));
                    idtemp = fix(rand(1,Ntemp)*Ntemp)+1 ;
                    tempDAT=[xtmp(idtemp), ytmp(idtemp), xytmp(idtemp)];
                else
                    Ntemp=prod(size(coef.value{j}));
                    idtemp = fix(rand(1,Ntemp)*Ntemp)+1 ;
                    tempDAT=coef.value{j}(idtemp);
                end
            else                
                if   EQUAL_LENGTH      
                    if use_dwt
                        Ntemp=floor(prod(size(tempDAT))/3);
                        Ntempref=prod(size(xtmp));
                        if Ntemp<Ntempref
                            idtemp = fix(rand(1,(Ntempref-Ntemp))*Ntempref)+1 ;
                            tempDAT= [tempDAT, xtmp(idtemp), ytmp(idtemp), xytmp(idtemp)];
                        elseif Ntemp>Ntempref
                            tempDAT=tempDAT(1:3*Ntempref);
                        end
                    else
                        Ntemp=prod(size(tempDAT));
                        Ntempref=prod(size(coef.value{j}));
                        if Ntemp<Ntempref
                            idtemp = fix(rand(1,(Ntempref-Ntemp))*Ntempref)+1 ;
                            tempDAT= [tempDAT, coef.value{j}(idtemp)];
                        elseif Ntemp>Ntempref
                            tempDAT=tempDAT(1:Ntempref);
                        end
                    end
                end                
            end  %if isempty(tempDAT);
            
            % Calculate BS Structure Functions
%             if ~fparam
%                 [T{bsid,:}]=FHandle(tempDAT);
%             else
                %[T{bsid,:}]=FHandle(tempDAT, param_est.param);
                [T{bsid,:}]=FHandle(tempDAT, param_est);
%             end

            if doB2;
                for bs2id=1:B2
                    tmpTT=cell(1,NOUT);
                    % get the coefficients at the indices for each resample
                    if use_dwt
                        tempDAT=[(DAT(j).X(squeeze(index2(bs2id,bsid,:)))), (DAT(j).Y(squeeze(index2(bs2id,bsid,:)))), (DAT(j).XY(squeeze(index2(bs2id,bsid,:)))), ];
                    else
                        tempDAT=(DAT(j).X(squeeze(index2(bs2id,bsid,:))));
                    end

                    % sort out NaNs
                    tempDAT=tempDAT(~isnan(tempDAT))';

                    % Catch case of empty resample (mainly last scale): use ordinary
                    % bootstrap sample
                    if isempty(tempDAT);
                        if use_dwt
                            Ntemp=prod(size(xtmp));
                            idtemp = fix(rand(1,Ntemp)*Ntemp)+1 ;
                            tempDAT=[xtmp(idtemp), ytmp(idtemp), xytmp(idtemp)];
                        else
                            Ntemp=prod(size(coef.value{j}));
                            idtemp = fix(rand(1,Ntemp)*Ntemp)+1 ;
                            tempDAT=coef.value{j}(idtemp);
                        end
                    else
                        if   EQUAL_LENGTH                      
                            if use_dwt
                                Ntemp=floor(prod(size(tempDAT))/3);
                                Ntempref=prod(size(xtmp));
                                if Ntemp<Ntempref
                                    idtemp = fix(rand(1,(Ntempref-Ntemp))*Ntempref)+1 ;
                                    tempDAT= [tempDAT, xtmp(idtemp), ytmp(idtemp), xytmp(idtemp)];
                                elseif Ntemp>Ntempref
                                    tempDAT=tempDAT(1:3*Ntempref);
                                end
                            else
                                Ntemp=prod(size(tempDAT));
                                Ntempref=prod(size(coef.value{j}));
                                if Ntemp<Ntempref
                                    idtemp = fix(rand(1,(Ntempref-Ntemp))*Ntempref)+1 ;
                                    tempDAT= [tempDAT, coef.value{j}(idtemp)];
                                elseif Ntemp>Ntempref
                                    tempDAT=tempDAT(1:Ntempref);
                                end
                            end
                        end                        
                    end

                    % Calculate BS Structure Functions
%                     if ~fparam
%                         [tmpTT{:}]=FHandle(tempDAT);
%                     else
                        %[tmpTT{:}]=FHandle(tempDAT, param_est.param);
                        [tmpTT{:}]=FHandle(tempDAT, param_est);
%                     end
                    tempTT(bs2id,bsid,:)=cell2mat(tmpTT);
                end
            end
        end

        estimates{j}.T=cell2mat(T);
        estimates{j}.stdt=std(estimates{j}.T);

        if doB2;
            estimates{j}.stdT=squeeze(std(tempTT));
            estimates{j}.TT=tempTT;
        end
    else
        estimates{j}.T=NaN(B1, length(estimates{j}.t));
        estimates{j}.stdt=NaN(1, length(estimates{j}.t));
        if doB2;
            estimates{j}.stdT=NaN(B1, length(estimates{j}.t));
            estimates{j}.TT=NaN(B2, B1, length(estimates{j}.t));
        end
    end
end
