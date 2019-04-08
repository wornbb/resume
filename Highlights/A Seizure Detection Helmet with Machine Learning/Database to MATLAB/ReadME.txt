The main purpose of this folder is to import EEG data from Physionet database to matlab.
This folder contains all necessary files to perform above function.
The stored data is abstracted in the following way:
	- sampling rates are uniformly 256 Hz. Hence, omitted.
	- the test duration, test start time, end time, patients name, etc... are omitted
	- the origin of the data (channel infomation) is completely kept.
		for each test there will be 32 channels, hence 32 data series for each test (.edf file)
		these stored data will has to be further selected. Because, our EEG chip only support 8 channels. And those particular 8 channels haven't been decided yet. 
		Hence, at this stage, the best I can do is to bookkeep everything here.

There are multiple .mat files that collectively maps the whole Physionet database. Each .mat file will have variables with same names. Hence, each .mat file needs to be processed individually. This is for the sake of memory space.

the data strucutre can be added here later, if necessary.
