It is strongly recommend to view "readme.docx" because some points can only be illustrated by graph.
Music Editor
It is a program that allows your write your own Music on Matlab and Play it¡­

1.	Install
(1.1)
I have provided a app package that allows you install it on the matlab and find it in MY APPS
 
And run it like a real program other than a m.file¡­.

(1.2)
If you wish, you can also run the Music_Editor_shen1_1386326.m
DO NOT DIRECTELY DOUBLE CLINK ¡° .fig¡± FILE TO RUN IT.
IT WILL NEVER WORK
2.	Background knowledge¡­
Well¡­If you want to write a music, you have to know how to write a note..
Matlab does not provide a function that allows me to show the melody like this..
 
So you have to write them in characters¡­

According to Wiki¡­
http://en.wikipedia.org/wiki/Note
And my simply modified.

You can find the note provided in the following table

3th octave	¡¡	¡¡	¡¡	¡¡	¡¡	¡¡	¡¡
¡¡	la	ti	do	re	mi	fa	so
  1/4 	a3q	b3q	c3q	d3q	e3q	f3q	g3q
  1/8 	a3e	b3e	c3e	d3e	e3e	f3e	g3e
  1/16	a3s	b3s	c3s	d3s	e3s	fs	g3s
4th octave	¡¡	¡¡	¡¡	¡¡	¡¡	¡¡	¡¡
¡¡	la	ti	do	re	mi	fa	so
  1/4 	a4q	b4q	c4q	d4q	e4q	f4q	g4q
  1/8 	a4e	b4e	c4e	d4e	e4e	f4e	g4e
  1/16	a4s	b4s	c4s	d4s	e4s	f4s	g4s
5th octave	¡¡	¡¡	¡¡	¡¡	¡¡	¡¡	¡¡
¡¡	la	ti	do	re	mi	fa	so
  1/4 	a5q	b5q	c5q	d5q	e5q	f5q	g5q
  1/8 	a5e	b5e	c5e	d5e	e5e	f5e	g5e
  1/16	a5s	b5s	c5s	d5s	e5s	f5s	g5s
							
3th octave							
	la	ti	do	re	mi	fa	so
  1/4 	a3q	b3q	c3q	d3q	e3q	f3q	g3q
  1/8 	a3e	b3e	c3e	d3e	e3e	f3e	g3e
  1/16	a3s	b3s	c3s	d3s	e3s	f3s	g3s
4th octave							
	la	ti	do	re	mi	fa	so
  1/4 	a4q	b4q	c4q	d4q	e4q	f4q	g4q
  1/8 	a4e	b4e	c4e	d4e	e4e	f4e	g4e
  1/16	a4s	b4s	c4s	d4s	e4s	f4s	g4s
5th octave							
	la	ti	do	re	mi	fa	so
  1/4 	a5q	b5q	c5q	d5q	e5q	f5q	g5q
  1/8 	a5e	b5e	c5e	d5e	e5e	f5e	g5e
  1/16	a5s	b5s	c5s	d5s	e5s	f5s	g5s
If you want to create a period of blank, in other word, pause for a few second.
Use blq to replace 1/4 note
Use ble to replace 1/8 note
Use bls to replace 1/16 note
Example
 
This melody only plays a 1/8 note do in 3th octave 
But before you hear that , it pauses for the time of 2 1/8 note.
3.	Grammar
If you are creating a new melody.
Start writing at the very beginning of the script.
Add a single space between every notes.eg:
 
If you are combining an ensemble
Just place the notes you write on the same vertical line.no more no less
Example
 
NOTE: in this situation there will be a space at the very beginning of each line
NOTE: NEVER ENTER A SAPCE AT THE END OF LINE, AND NEVER LEAVE A BLANK WHEN THERE SHOULD BE A NOTE
Error presentation:
(1)	Leave a blank
 
(2)	Not in vertical line
 

4.	I have provided a sample melody (Ode_to_joy.m)  created by this program
So if you don¡¯t know how to test, this might help.
To listen this melody
1)	Run the program
2)	Enter the name in the edit text above the ¡°play¡± button
