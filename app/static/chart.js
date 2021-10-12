var Chart = function(opts) {
    // load in arguments from config object
    this.data = opts.data
    this.element = opts.element;
    this.maxLength = opts.maxLength;
    console.log(this.data)
    if (this.data.I.length>this.maxLength) {this.maxLength = this.data.I.length}
    
    // create the chart
    this.colorCodeState = {"S": "green", "E":"orange" ,"I": "red", "R": "blue", "G":"black" , "O": "purple"}
   // this.colorCodeState = {"I": "red", "E":"orange","R": "blue"}
   // this.colorCodeState = {"I": "red" }
    this.labelCodeState = {"S": "Susceptible", "E":"Exposed" ,"I": "Infected", "R": "Recovered", "G":"Superspreader" , "O": "Seed"}

    this.draw();
}

Chart.prototype.draw = function() {
    
    // define width, height and margin
    this.width = this.element.offsetWidth;
 //  this.width = 560;
    this.height = this.width / 2;
    this.margin = {
        top: 20,
        right: 75,
        bottom: 45,
        left: 50
    };
    
    // set up parent element and SVG
    this.element.innerHTML = '';
    var svg = d3.select(this.element).append('svg');
    svg.attr('width',  this.width);
    svg.attr('height', this.height);

    // we'll actually be appending to a <g> element
    this.plot = svg.append('g')
        .attr('transform','translate('+this.margin.left+','+this.margin.top+')');
    
    // create the other stuff
    this.createScales();
    this.addAxes();

    legColors= []
    legStrings = []
    for (key in this.colorCodeState ) {
        if (key in this.data) {
		    this.addLine(this.data[key],this.colorCodeState[key],1,2);
		    legStrings.push(this.labelCodeState[key])
		    legColors.push(this.colorCodeState[key])

       }
    }
    
    
    var ordinal = d3.scaleOrdinal()
	  .domain(legStrings)
	  .range(legColors);
	
	var translateString = "translate("+ (this.width*2/3).toString(10)+",20)";
	this.plot.append("g")
	  .attr("class", "legendOrdinal")
	  .attr("transform", translateString);
	
	var legendOrdinal = d3.legendColor()
	  //d3 symbol creates a path-string, for example
	  //"M0,-8.059274488676564L9.306048591020996,
	  //8.059274488676564 -9.306048591020996,8.059274488676564Z"
	  .shape("path", d3.symbol().type(d3.symbolCircle).size(100)())
	  .shapePadding(5)
	  //use cellFilter to hide the "e" cell
	  .cellFilter(function(d){ return d.label !== "e" })
	  .scale(ordinal);
	
	svg.select(".legendOrdinal")
	  .call(legendOrdinal);    
}

Chart.prototype.createScales = function(){
    
    // shorthand to save typing later
    var m = this.margin;
    
    this.xScale = d3.scaleLinear()
        .range([0, this.width-m.right])
        .domain([0,this.maxLength]);
    if ("S" in this.colorCodeState) {
        this.yScale = d3.scaleLinear()
        .range([this.height-(m.top+m.bottom), 0])
          .domain([0,Math.max.apply(false, this.data.S)*1.2]);
    }
    else if ("R" in this.colorCodeState) {
        this.yScale = d3.scaleLinear()
        .range([this.height-(m.top+m.bottom), 0])
          .domain([0,Math.max.apply(false, this.data.R)*1.2]);
    }
    else {
        this.yScale = d3.scaleLinear()
        .range([this.height-(m.top+m.bottom), 0])
                  .domain([0,Math.max.apply(false, this.data.I)*1.2])
    }


}

Chart.prototype.addAxes = function(){
    var m = this.margin;

    // create and append axis elements
    // this is all pretty straightforward D3 stuff
    var xAxis = d3.axisBottom(this.xScale)
    
    var yAxis = d3.axisLeft(this.yScale)
    
    this.plot.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + (this.height-(m.top+m.bottom)) + ")")
        .call(xAxis);
    
    this.plot.append("g")
        .attr("class", "y axis")
        .call(yAxis)
}

Chart.prototype.addLine = function(data,color,stroke_opacity,stroke_width){
    // need to load `this` into `_this`...    
    var _this = this;
    var line = d3.line()
    	.x(function(d, i) { return _this.xScale(i); })
    	.y(function(d) { return _this.yScale(d); })
    	 
    this.plot.append('path')
        // use data stored in `this`
        .datum(data)
        .attr('class','line')
        .attr('d',line)
        // set stroke to specified color, or default to red
        .style('stroke', color)
        .style('stroke-opacity', stroke_opacity)
        .style('stroke-width', stroke_width)
        .style("fill","none");
}
