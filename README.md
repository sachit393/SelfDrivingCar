# SelfDrivingCar
Self Driving Car implemented as a Part of Artificial Intelligence course. 
<ul>
<br>
<li>The input states are passed to a Deep Q Network and output is the action that car has to take</li>
<br>
<li>Input is a tuple consisting of the density of sand around the car and the orientation of car's velocity relative to its destination. 
Car is given negative reward for passing over sand, going away from destination and going too closed to boundary and positive reward for getting closer to destination.</li>
<br>
<li>Temporal difference is used to calculate loss function by back-propagating through the Neural Network and updating weights and bias</li>
</ul>
<br>
<br>
<p>KIVY library was used for GUI(Copy pasted most of the KIVY based part ;) )
