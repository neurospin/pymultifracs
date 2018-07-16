function [estimates]=resample2d(coef, param_bs, param_est, Circular);
% function [estimates]=resample2d(coef, param_bs, param_est, Circular);
%
% Herwig Wendt, Lyon, 2006 - 2008

% p = param_est.p;

%-InputError1='param_bs must be a structure with elements:\n   param_bs.n_resamp_1 \n   param_bs.n_resamp_2\n   param_bs.block_size\n   param_bs.ci_method\n   param_bs.verb (optional)\n';
%-try  NB1=isreal(param_bs.n_resamp_1);      if ~NB1; param_bs.n_resamp_1=999; end;  catch error(InputError1, NB1); end
%-try  NB2=isreal(param_bs.n_resamp_2);      if ~NB2; param_bs.n_resamp_2=50;  end; catch error(InputError1, NB2); end
%-try  NB3=isreal(param_bs.block_size);   if ~NB3; param_bs.blocklength=1; end;  catch error(InputError1, NB3); end
%-try  NB4=isreal(param_bs.ci_method);        if ~NB4; param_bs.ci_method=[1:6];end;  catch error(InputError1, NB4); end
%-try CHR=ischar(param_est.fhandle); catch error('The structure param_est must contain a field  param_est.fhandle with a valid function handle'); end
%-if ~CHR; error('The function handle param_est.fhandle is not valid'); end

FHandle=param_est.fhandle;   % Estimator name

%FHandle=str2func(fhandle);       % Function handle for Estimator
%NOUT=nargout(FHandle);           % Number of Output Arguments of Estimator

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

% need double bootstrap values for Adjusted Methods
if STU|PERADJ|BASADJ; doB2=1; else; doB2=0; end
% Bootstrap parameters
B1=param_bs.n_resamp_1;        % number of primary bootstrap resamples
B2=param_bs.n_resamp_2;        % number of bootstrap resamples for variance estimates (used for Normal, Studentised, Variance Stabilising Transformation)
Block=param_bs.block_size;  % Block length for moving blocks resampling

if length(Block)>1; BlockX=Block(1); BlockY=Block(2); else; BlockX=Block; BlockY=Block; end
BLOCK=max(Block);

try 
    Circular; 
catch; 
    Circular=1; 
end;

% Determine is input is wavelet coefficients or leaders
if ndims (coef) == 3
    use_dwt = 1;
else
    use_dwt = 0;
end
% try 
%     coef.value; 
%     use_dwt=0;     % Leaders
% catch
%     use_dwt=1;     % Coefficients don't have field "value"
% end

if use_dwt
    X  = coef(:, :, 1);
    Y  = coef(:, :, 2);
    XY = coef(:, :, 3);
    [Nx, Ny]=size(X);
    %sample=[reshape(X,1,[]),reshape(Y,1,[]),reshape(XY,1,[])];
    %sample=coef.vector.all;
else
%     leader=max(coef.value.x, max(coef.value.y, coef.value.xy));
    %leader=coef.vector.max;
    [Nx, Ny]=size(coef);
    %sample=reshape(leader,1,[]);
    %sample=coef.vector.max;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    ESTIMATE AND BOOTSTRAP ESTIMATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate estimate
%%%%%%%%%%%%%%%%%%%%%%%%%%
%t=cell(1,NOUT);
% if ~fparam
%     %[t_out{:}]=feval(fhandle,X);
%     [t{:}]=FHandle(sample);
% else
    %[t{:}]=FHandle(sample,param_est.param);
%    [t{:}]=FHandle(sample,param_est);
% end
% get length of estimates
% for ii=1:NOUT
%     LEst(ii)=length(t{ii});
% end
%t=cell2mat(t);

% keyboard

%if NOR|BAS|PER|STU|BASADJ|PERADJ
T=cell(B1,NOUT);
if STU|BASADJ|PERADJ
    TTT=cell(B2,NOUT);
end

%% CREATE BLOCKS OF IDs
if BLOCK>1
    try % fast but memory intensive
        %- initialize blocks of indices
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
          %- ADAPT OLD CODE HERE
        error('Doing Blocks --> Out of memory: Adapt old method (HW 25/12/06, St. Ulrich)');
    end

    % INITIALIZE BLOCK BOOTSTRAP
    if ~Circular;
        N_rangex=max(Nx-BlockX+1,1);
        N_rangey=max(Ny-BlockY+1,1);
    else
        N_rangex=Nx;
        N_rangey=Ny;
    end
    N_BSx=ceil(Nx/BlockX);
    N_BSy=ceil(Ny/BlockY);
    N_resamplex=N_BSx*BlockX;
    N_resampley=N_BSy*BlockY;
end

% RESAMPLING
for b=1:B1
    if BLOCK==1
        index = fix(rand(1,Nx*Ny)*Nx*Ny)+1 ;
        if use_dwt;
            Xsample=X(index);
            Ysample=Y(index);
            XYsample=XY(index);
            resample=[Xsample, Ysample, XYsample];
        else
            resample=coef(index);
        end
    else  % block bootstrap
        index  = fix(rand(1,N_BSx*N_BSy)*N_rangex*N_rangey)+1;
        %- !!!! shouldn't Nx be N_rangex here (when not circular) (HW, 30/01/2007, Lyon) :
        indexx = rem(index-1,Nx)+1        ;
        indexy = ceil(index/Nx) ;    % block column index
        if ~Circular
            tmpidx=find(indexx>N_rangex); indexx(tmpidx) = N_rangex; % block row index
            tmpidy=find(indexy>N_rangey); indexy(tmpidy) = N_rangey; % block row index
        end
        %- calculate sample indeces from block index
        tempBX=repmat(BX(:,indexx),BlockY,1);
        tempidx=reshape(tempBX,1,[]);
        tempBY=repmat(reshape(BY(:,indexy),[],1),1,BlockX);
        tempidy=reshape(tempBY',[],1)';
        tempid=tempidx+(tempidy-1)*Nx;

        if use_dwt
            Xsample=X(tempid);
            Ysample=Y(tempid);
            XYsample=XY(tempid);
            resample=[Xsample, Ysample, XYsample];
        else
            resample=coef(tempid);
        end           
    end
    [T{b,:}]=FHandle(resample, param_est);
    if doB2
        for bb=1:B2
            if Block==1
                index2 = index(fix(rand(1,length(index))*length(index))+1) ;
                if use_dwt;
                    Xresample=X(index2);
                    Yresample=Y(index2);
                    XYresample=XY(index2);
                    reresample=[Xresample, Yresample, XYresample];
                else
                    reresample=coef(index2);
                end
            else % block bootstrap
                index2  = index(fix(rand(1,length(index))*length(index))+1);
                %- !!!! shouldn't Nx be N_rangex here (when not circular) (HW, 30/01/2007, Lyon) :
                indexx2 = rem(index2-1,Nx)+1        ;
                indexy2 = ceil(index2/Nx) ;    % block column index
                if ~Circular
                    tmpidx2=find(indexx2>N_rangex); indexx2(tmpidx2) = N_rangex; % block row index
                    tmpidy2=find(indexy2>N_rangey); indexy2(tmpidy2) = N_rangey; % block row index
                end
                %- calculate sample indeces from block index
                tempBX2=repmat(BX(:,indexx2),BlockY,1);
                tempidx2=reshape(tempBX2,1,[]);
                tempBY2=repmat(reshape(BY(:,indexy2),[],1),1,BlockX);
                tempidy2=reshape(tempBY2',[],1)';
                tempid2=tempidx2+(tempidy2-1)*Nx;
                if use_dwt
                    Xresample=X(tempid2);
                    Yresample=Y(tempid2);
                    XYresample=XY(tempid2);
                    reresample=[Xresample, Yresample, XYresample];
                else
                    reresample=coef(tempid2);
                end

                [TTT{bb,:}]=FHandle(reresample, param_est);

            end                
        end
        stdT(b,:)=std(cell2mat(TTT)); % WORKS
        TT(:,b,:)=cell2mat(TTT);
    end
end
T=cell2mat(T);
stdt=std(T);

if ~(STU)
    stdT=NaN(1, B1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save bootstrap estimates and distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%
estimates.T = T;
estimates.stdt = stdt;
if doB2
    estimates.TT = TT;
    estimates.stdT = stdT;
end

