function varargout = rope_GUI(varargin)
% ROPE_GUI MATLAB code for rope_GUI.fig
%      ROPE_GUI, by itself, creates a new ROPE_GUI or raises the existing
%      singleton*.
%
%      H = ROPE_GUI returns the handle to a new ROPE_GUI or the handle to
%      the existing singleton*.
%
%      ROPE_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ROPE_GUI.M with the given input arguments.
%
%      ROPE_GUI('Property','Value',...) creates a new ROPE_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before rope_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to rope_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help rope_GUI

% Last Modified by GUIDE v2.5 12-Jun-2019 23:00:02

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @rope_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @rope_GUI_OutputFcn, ...
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


% --- Executes just before rope_GUI is made visible.
function rope_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to rope_GUI (see VARARGIN)

% Choose default command line output for rope_GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes rope_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = rope_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

global x t sineSum han

ii=round(get(hObject,'Value') * numel(t) ); 
ii=max(ii,1); ii=min(ii,numel(t)); 


plot(han,x,sineSum(:,ii))

ylim([-1 1])

% if 0 
% figure 
% surf(squeeze(dd),'EdgeColor','none') 
% end 






% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over slider1.
function slider1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% get(hObject,'Value')



