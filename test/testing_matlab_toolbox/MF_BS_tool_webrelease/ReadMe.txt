COPYRIGHT AND CONDITIONS OF USE The codes are made freely available for non-commercial use only, specifically for research and teaching purposes, provided that the copyright headers at the head of each file are not removed, and suitable citation is made in published work using the routine(s). No responsibility is taken for any errors in the code. The headers in the files contain copyright dates and authors. In addition there is a copyright on the collection as made available here (September 2008).

QUICK START
To install the toolbox, copy it to your desired destination and add the path to the main folder "WLBMF_tool" to your MATLAB search paths. Executing "pathMF_BS_toolbox.m" will add the paths to all subfolders.
In order to run the demo files ("demo_MF_BS_tool.m" and "demo_MF_BS_test_constancy.m"), you will need to download the example data available on my web site. Alternatively, you can of course use your own data.

Overall, the demo scripts provide all information necessary for the use of the toolbox. You can also very easily modify these scripts to do analysis on your own data and adapt them to your problems. 
Yet, you will need basic understanding on wavelet-based multifractal analysis to achieve relevant results.

Following are some complementary notes:

USAGE OF "MF_BS_tool.m"
- "MF_BS_tool.m" is the front-end function and gives access to all functionalities for both 1d signals and 2d images. Input parameters, output and usage are commented in detail in "demo_MF_BS_tool.m".
- Switch between 1d and 2d is automatic. Analysis of 2d images is limited to images whose size is a power of 2 (this is a requirement of the RICE wavelet toolbox)
- Input and output variables are heavily based on structures due to the important number of parameters and estimates in wavelet based multifractal analysis. The convenience function "MF_BS_tool_param.m" helps you writing all your input parameters in appropriate structures.
- You can reuse structure functions that you previously calculated if you want to change estimation parameters that do not concern the underlying wavelet transform and the number of bootstrap resamples. Simply input the structure functions instead of the data (cf. bottom of demo file). This will be particularly useful when data size is large and bootstrap is used (wavelet transform and bootstrap resampling step are skipped, hence reducing significantly computational cost). The interactive mode also does precisely that.
FURTHER COMMENTS:
- If you do not want to use the bootstrap functionalities, simply set the parameter for number of resamples to zero.
- A relevant multifractal analysis critically depends on the selection of an appropriate scaling range and hence visual inspection of structure functions. MF_BS_tool groups structure functions (for all scaling exponents and log-cumulants) in a single figure as subplots, which can become very small. 
You can enlarge any subfigure by simply clicking on it. The same is true for plots of estimates.