function varargout = Music_Editor_shen1_1386326(varargin)
% MUSIC_EDITOR_SHEN1_1386326 MATLAB code for Music_Editor_shen1_1386326.fig
%      MUSIC_EDITOR_SHEN1_1386326, by itself, creates a new MUSIC_EDITOR_SHEN1_1386326 or raises the existing
%      singleton*.
%
%      H = MUSIC_EDITOR_SHEN1_1386326 returns the handle to a new MUSIC_EDITOR_SHEN1_1386326 or the handle to
%      the existing singleton*.
%
%      MUSIC_EDITOR_SHEN1_1386326('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MUSIC_EDITOR_SHEN1_1386326.M with the given input arguments.
%
%      MUSIC_EDITOR_SHEN1_1386326('Property','Value',...) creates a new MUSIC_EDITOR_SHEN1_1386326 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Music_Editor_shen1_1386326_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Music_Editor_shen1_1386326_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Music_Editor_shen1_1386326

% Last Modified by GUIDE v2.5 28-Mar-2014 02:43:59

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Music_Editor_shen1_1386326_OpeningFcn, ...
                   'gui_OutputFcn',  @Music_Editor_shen1_1386326_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Music_Editor_shen1_1386326 is made visible.
function Music_Editor_shen1_1386326_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Music_Editor_shen1_1386326 (see VARARGIN)

% Choose default command line output for Music_Editor_shen1_1386326
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Music_Editor_shen1_1386326 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Music_Editor_shen1_1386326_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
%play function
load violin
name=get(handles.Music,'string');
%check whether the script is a ensemble
choice=menu('Is this an ensemble ?','no','yes');
switch choice
    case 1    %if not a ensemble,play it directly.
    m=readmelody(name);
    m=ryhthm(m);
    sound(m,44100);
    case 2    %now we have some problem.......
        fid=fopen(name);
        %input test,sometimes, people like me forget .m....damn it
    if fid == -1
        name=[name '.m'];
        fid=fopen(name);
    end
%formating melody
%Have to catch the note of the same melody and sort them.
    melodychar=[];
    %fristly read the all the melody as a whole
    while feof(fid) == 0 
        m=fgetl(fid);
        %Diffrence (selfreminder)
        melodychar=[melodychar,m];
    end
    fclose('all');
    %I mean 'complex' is a really complex combination........
    complex=melodychar;
    number=inputdlg('How many melodies did you combined including the main melody ?');
    %maybe some people think 'char' is a number ...Anyway..I won't let you
    %do this
    while isempty(str2num(number{1,:}))
            number=inputdlg('Please Enter a number');
    end
    % I cannot believe that inputdlg will give me a
    % string!!!!(selfreminder)
    number=str2num(number{1,:});
    submelody=cell(1,number);
    for i=1:number
        for n=1:size(complex,2)/(20*number)
            %"catch" equation...now,you see the complexity.Actually it`s
            %not that hard 2333333
        tagg=complex(1,1+(i-1)*20+20*(n-1)*number:20+(i-1)*20+20*(n-1)*number);
        submelody{1,i}=[submelody{1,i} tagg];
        end
        submelody{1,i}=ryhthm(submelody{1,i});
    end 
    %finally here we go...
    for i=1:number
        sound(submelody{1,i},44100);
    end
end

% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton2.
%in the end
%number is a number
function pushbutton2_Callback(hObject, eventdata, handles)
choice = menu('Choose What you want to do','Create New Melody'...
,'Edit Melody','Combine Melodies to an Ensemble') ;
%maybe some global varialbes are unnecessary,just forget them...the
%structure of this function had changed dramatically,hence it`s kinda
%historical problem...
global number filename mainmelody
switch choice
    case 1  %so easy....write the melody in m file
        edit %newmelody maybe 'untiled' will push the editor give it a name?
    case 2  %maybe you want to change your script...IDK...
        file_name = inputdlg('Melody Name you want to edit');
        file_name = char(file_name);
        edit(file_name)
    case 3  %OK this part is torturing...
        first=menu('Have you written the main melody ?','Yes','No');
        %check whether have main melody
        %for this 'if'
        %Firstly check whether you have finish the first part of ensemble
        %If no. ...then go and write one
        if first == 2
            newmelody = melodyname('Creating a New melody,Please enter the name');
            %name test
             while isempty(str2num(newmelody(1,1)))~=1
                newmelody=melodyname('Please Input a name with a character initial');
             end
            edit(newmelody);
            %nice reminder to the user...
            menu('Create the main melody then try again','ok');
            return
        else
            %if yes ....then we have a lot of work to do...
            mainmelody=melodyname('input the name of your main melody');
            %the name of mainmelody ,used to open the file later
            %on(selfreminder)
            fid=fopen(mainmelody);
            %dumb prevention
        if fid == -1 
            mainmelody=[mainmelody '.m'];
        end
         fclose('all');
            mainmelody=readmelody(mainmelody);
        end
        %duet or trio or ..sorry for my poor vocabulary...
        number = inputdlg('How many Melodies do you want to combine?');
        %number test
        while isempty(str2num(number{1,:}))
            number=inputdlg('Please Enter a number');
        end
        number =str2double(number);
        submelody =melodyname('Input the name of your submelody');
        %name test
        while isempty(str2num(submelody(1,1)))~=1
            submelody=melodyname('Please Input a name with a character initial');
        end
        %well....I know what you are thinking..
        %but it`s the best way i can figure out...
        %and viriable has told what i want to do
        reformat='%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c';
        p='\n';
        for i=2:number
            p=[p '\n'];
        end
        reformat=[reformat p];
        %check .m
            fid=fopen(submelody);
        if fid == -1 
            submelody=[submelody '.m'];
        end
        fclose('all');
        filename=submelody;% '.m'];
        fid=fopen(filename,'w');
        fprintf(fid,reformat,mainmelody);
        fclose('all');
        edit(filename)
end
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
function [newmelodyname]=melodyname(inputt)
inputt=char(inputt);
newmelodyname = inputdlg(inputt);
newmelodyname = char(newmelodyname);

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
%as a musician,you may want a  metronome
global num bob body x0 y0
try
    delete(bob,body)
catch
end
%get the 'frequency'(beat per min)
text=get(handles.text,'string');
num=str2num(text);
distance=(num/140)*(pi/18);
%delete original metronome
try
    delete(handles.h2,handles.h3)
catch
end
%above /functions are used to reset the graph
%reset metronome(selfreminder)
theta0=pi/9;
x0=0.7*sin(theta0);
y0=(-1)*0.7*cos(theta0);
bob=line(x0,y0,'Marker','o');
%I mean.the string....
body=line([0;x0],[0;y0],'linestyle','-',...
    'erasemod','xor');
t=0;
T=120/num;
%period of metronome
dt=0.2*pi;
global a
%little swich..cute..
a=1;
interval=1:150;
ti=interval.*sin(interval);
while a ==1
    t=t+dt;
    theta=theta0*cos(t);
    x=0.7*sin(theta);
    y=(-1)*0.7*cos(theta);
    try
        set(bob,'xdata',x,'ydata',y);
        set(body,'xdata',[0;x],'ydata',[0;y]);
    catch
    end
    drawnow;
    pause(T/10);
    if cos(t)==1||cos(t)==-1
        sound(ti,13000)
    end
end
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
global a
a=0;
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function text_Callback(hObject, eventdata, handles)
% hObject    handle to text (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of text as text
%        str2double(get(hObject,'String')) returns contents of text as a double


% --- Executes during object creation, after setting all properties.
function text_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over text.
function text_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to text (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function clock_CreateFcn(hObject, eventdata, handles)
axis('off');
%topaxle=[0,0];
%top
hold on
h1=line(0,0,'Marker','.','MarkerSize',15);
handles.h1=h1;
%body
h2=line([0;0],[-0.7;0],'Marker','-');
handles.h2=h2;
%bob
%'erasemode','xor'
h3=line(0,-0.7,'Marker','o',...
'markersize',10);
handles.h3=h3;
guidata(hObject,handles);
axis([-0.75,0.75,-1.25,0]);
% hObject    handle to clock (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate clock


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over pushbutton4.
function pushbutton4_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
global a
a=0;
close all force
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over pushbutton5.
function pushbutton5_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
global a bob body x0 y0 
try
    delete(bob)%,'color','w')
    delete(body)%,'color','w')
catch
end
a=0;
x0=0;
y0=-0.7;
bob=line(x0,y0,'Marker','o');
body=line([0;x0],[0;y0],'linestyle','-',...
    'erasemod','xor');
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on key press with focus on figure1 and none of its controls.
function figure1_KeyPressFcn(hObject, eventdata, handles)
%load(a,jpg)
% hObject    handle to figure1 (see GCBO)
% eventdata  structure with the following fields (see FIGURE)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function axes2_CreateFcn(hObject, eventdata, handles)

% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes2



function Music_Callback(hObject, eventdata, handles)
% hObject    handle to Music (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Music as text
%        str2double(get(hObject,'String')) returns contents of Music as a double


% --- Executes during object creation, after setting all properties.
function Music_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Music (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%might be used several times....
function [melodychar]=readmelody(name)
%in case the input of name is incomptlete
fid=fopen(name);
if fid == -1 
    name=[name '.m'];
    fid=fopen(name);
end
%formating melody
blank=' ';
melodychar=[];
while feof(fid) == 0 
    m=fgetl(fid);
    melodychar=[melodychar,blank,m];
end
fclose('all');

function [ryhthm]=ryhthm(melodychar)
load violin
length=size(melodychar);
melody=[];
for i=1:length(1,2)/4
    transitionchar=melodychar(1,2+4*(i-1):4+4*(i-1));
    transitioneval=eval(eval('transitionchar'));
    melody=[melody transitioneval];
end
%play sound
ryhthm=melody;
%sound(melody,44100);


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over pushbutton1.
function pushbutton1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object deletion, before destroying properties.
function clock_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to clock (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on mouse press over axes background.
function clock_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to clock (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on key press with focus on pushbutton5 and none of its controls.
function pushbutton5_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  structure with the following fields (see UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function pushbutton5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
