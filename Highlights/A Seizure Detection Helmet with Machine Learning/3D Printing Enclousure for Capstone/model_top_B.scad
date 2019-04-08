$fn = 30;
thou2mm = 0.0254;


difference(){
    translate([1.5,1.5,0]){cube([64,64,1.5]);};
    translate([1.5,1.5,-10]){
    translate([129*thou2mm,805*thou2mm]) {
        cube([4.5,4.5,50]);
    }
    translate([129*thou2mm,1605*thou2mm]){
        cube([4.5,4.5,50]);
    }
    translate([2292*thou2mm,805*thou2mm]){ 
        cube([4.5,4.5,50]);
    }
    translate([2292*thou2mm,1605*thou2mm]){ 
        cube([4.5,4.5,50]);
    }
}
}