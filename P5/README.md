

<h3><b>Baseball data visualization using D3 and Dimple.js</b></h3>

<b>Summary</b> 

Baseball data from the Dataset options provided by Udacity was used for project. Initial exploratory analysis of the data showed a relationship between batting average and home runs. The batting average did not increase with increase in the number of home runs hit, it seemed to taper off at 0.3. Exploring the data further across handedness of the players, the average home runs scored seemed to be slightly higher for left handed players compared to others.

I landed on two insights from the data 
<b>One:</b> The dimished returns relationship between home runs hit as can be seen from the plot below. 
<b>Two:</b> The higher average home runs hits by left handed players compared to the other. 
I chose to go with highlighting finding number two for this project. After exploring several visualization choices, I went with a bar plot which best described the realtionship between average home runs and handedness of the players.    

![Home Runs vs Batting Average](/P5/HR_vs_avg.jpeg?raw=true)

![Home Runs vs Batting Average](/P5/Handedness_average.jpeg?raw=true)

 
I focused my visualization on highlighting this difference. Initially I went with a line plot for each group using series interpolation on Dimple.js. I chose the smoothing technique that best represented the peaks of home runs scored by left handed players which contributed the higher average. The smoother “bundle” gave results closer to what I was expecting to see.  Removed the plotted points that were hidden but got highlighted on an accidental mouse-over.  
The initial visualization the x-axis labeling was busy as I was using batting average “avg” as a category, I fixed that by using an alternate x-axis and hiding the plotted one. 
I added title so the visualization and adjusted its position on the graph
 
![Initial Visualization](/P5/Initial_visualization.png?raw=true) 

I reconsidered the initial design decision of using line charts for showing the average home runs.  The chart showed the peaks for left handed players were higher than the right handed players, red and blue respectively. It also clearly showed the least average among all were both handed players.  I decided to get feedback for this visualization to see if this was good enough. 

<b>Feedback</b> 

I showed the chart to three people and asked for their feedback and specific comments. I explained what I was trying to accomplish with the visualization, the data that was being used for this purpose and the approached I had considered until then. The following are the main points of their feedback: 

1.	The peak of the red lines were visible. The axis is clear for interpretation. 
The title is not coherent with the visualization. Overall it a clean graph, I see “L” wins on home runs here. 

2.	It looks a bit busy to see all the lines at once. I clicked on the peak and expected to see the point statistic. Wonder if can highlight area under the lines for each group, that might be interesting to look at 

3.	I think it would be nice to if I can get some text on hover, such as line legend. Upon closer observation I can the trends for the batsmen. The x-axis is a bit confusing to read. 

<b> Post Feedback changes</b>

I incorporated the following changes to the visualization post feedback: 

1.	I changed the plot from line to area plot which is more visually rendering than the latter, given the findings I wanted to highlight. 
2.	Added a mouseover event to highlight the area under each group 
3.	Changed the title to better suit the finding communicated in the visualization 
4.	Corrected the x-axis ticks so they are ordered evenly.  
5.	Changed the position of the legend so it’s closer to the plots to lookup to for reference 
The post feedback version of the plot if as below: 

![Final Visualization](/P5/Final_visualization.png?raw=true)
    
<b>Further Changes to the Chart</b> 

After further feedback, on a comparison between the area plot and a bar plot, I found that a bar plot was more visually explanatory of the insight I was trying to drive home. I made the following additional changes: 

1. Changed the plot type to bar 

2. Removed the "avg" axis to compare only between average home runs between different handedness 

3. Added barcharttooltip to highlight the bar stats to make the chart more interactive 

4. Expanded the legend abbreviations to full form 

The final chart is as below: 


<b>References:</b> 

Dimplejs.org

https://github.com/PMSI-AlignAlytics/dimple/wiki/

Stackoverflow.com 
