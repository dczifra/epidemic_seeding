var BarPlot = function(opts) {
    // load in arguments from config object
    this.data = {}
    for (key in opts.data) {
    	this.data[parseFloat(key)]=opts.data[key]
    }
    
    this.element = opts.element;
    this.maxLength = Math.max.apply(false, Object.keys(this.data));
    if (opts.maxLength>this.maxLength) {
        this.maxLength=opts.maxLength;
    }
    console.log(this.data)
    
    this.draw();
}

BarPlot.prototype.draw = function() {
    
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
    this.plotHeight=this.height-(this.margin.top+this.margin.bottom)
    orderedKeys = Object.keys(this.data).sort(function(a,b){return a-b})
    this.barWidth = orderedKeys[1]-orderedKeys[0]
    
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
    
    console.log(this.xScale(this.barWidth))
    console.log(Object.keys(this.data).map(this.xScale))
    
    this.plot = svg.append('g')
        .attr('transform','translate('+this.margin.left+','+this.margin.top+')');

    
    _this = this 
    this.plot.selectAll("rect")
      .data(Object.keys(this.data))
    .enter().append("rect")
      .style("fill", "steelblue")
      .attr("x", function(d) { return _this.xScale(d); })
      .attr("width", _this.xScale(_this.barWidth))
      .attr("y", function(d) { return _this.yScale(_this.data[d]); })
      .attr("height", function(d) { return _this.plotHeight - _this.yScale(_this.data[d]); });
	
}

BarPlot.prototype.createScales = function(){
    
    // shorthand to save typing later
    var m = this.margin;
    
    this.xScale = d3.scaleLinear()
        .range([0, this.width-m.right])
        .domain([0,this.maxLength]);
    this.yScale = d3.scaleLinear()
        .range([this.height-(m.top+m.bottom), 0])
        .domain([0,Math.max.apply(false, Object.values(this.data))*1.2]);


}

BarPlot.prototype.addAxes = function(){
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

