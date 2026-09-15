/* Local visual QA only: static frontend with synthetic API fixtures. No DB,
   model, auth credentials or production traffic. Student :8123, instructor :8124. */
const http=require("node:http"),fs=require("node:fs"),path=require("node:path");
const graph=require("./graph-fixture.cjs");
const frontend=path.resolve(__dirname,"../../frontend"),counts={};
const assignments=[{id:"lab1",name:"LAB1 – Dictionaries",ready:3,total:3,published:true,created_at:"2026-09-10T12:00:00Z"},{id:"hw3",name:"HW3 – Stacks and Calculators",ready:11,total:12,published:true,created_at:"2026-09-08T12:00:00Z"}];
const problem={slug:"invert",title:"Invert",assignment_id:"lab1",description:"Swap the keys and the values of d, keeping only the unambiguous ones.\n\nA value that appears exactly once becomes a key in the result, paired with the key it came from. Leave repeated values out of the result.\n\n>>> invert({'one': 1, 'two': 2, 'three': 3})\n{1: 'one', 2: 'two', 3: 'three'}\n>>> invert({'one': 1, 'uno': 1, 'three': 3})\n{3: 'three'}\n>>> invert({})\n{}",ready:true,status:"progress",solved:1,total:3,percent:33,steps:[{number:1,status:"passed",attempts:1},{number:2,status:"needs_work",attempts:2},{number:3,status:"not_started",attempts:0}],last_activity:new Date().toISOString()};
const problems=[problem,{...problem,slug:"is-empty",title:"isEmpty",assignment_id:"hw3",status:"solved",solved:3,percent:100},{...problem,slug:"length",title:"__len__",assignment_id:"hw3"}];
function progress(){
 const now=new Date();
 return {generated_at:now.toISOString(),summary:{problems:14,completed:2,independent:2,in_progress:3,not_started:9,total:32,earned:5,shown:0,remaining:27,percent:16,completion_percent:14,ungraded_problems:0,assignments:2,completed_assignments:0,active_days:3,weekly_submissions:9},
 assignments:assignments.map(a=>({...a,problems:a.ready,completed:1,in_progress:a.id==="lab1"?1:2,total:16,earned:3,percent:19,completion_percent:Math.round(100/a.ready)})),problems,
 activity:Array.from({length:7},(_,i)=>({date:new Date(+now-(6-i)*86400000).toISOString().slice(0,10),submissions:[0,2,0,4,0,0,3][i],passed:[0,1,0,2,0,0,2][i],active:[1,3,6].includes(i)})),
 recent:[["opened","invert","Invert"],["completed","is-empty","isEmpty"],["passed","is-empty","isEmpty"],["practiced","length","__len__"]].map(([kind,slug,title],i)=>({kind,slug,title,assignment_id:slug==="invert"?"lab1":"hw3",step:1,at:new Date(+now-(i+1)*90000).toISOString()})),next_up:["invert","length"]};
}
function serve(role,port){
 http.createServer((req,res)=>{
  const url=new URL(req.url,"http://localhost"),p=url.pathname;
  const send=(data,status=200)=>{res.writeHead(status,{"Content-Type":"application/json","Cache-Control":"no-store"});res.end(JSON.stringify(data));};
  if(p==="/__requests")return send(counts);
  if(!/\.(html|js|css|svg)$/.test(p)&&p!=="/"){
    counts[role+" "+p]=(counts[role+" "+p]||0)+1;
    if(p==="/auth/me")return send({name:"Alex Morgan",student_id:"visual-"+role,role});
    if(p==="/assignments")return send({assignments});
    if(p==="/student/progress")return setTimeout(()=>send(progress()),250);
    if(p==="/solved")return send({slugs:["is-empty"],opened:["invert","length"],last_slug:"invert"});
    if(/^\/assignments\/[^/]+\/problems$/.test(p))return send({problems:problems.filter(x=>x.assignment_id===p.split("/")[2])});
    if(p==="/decompose_chunks")return send({session_id:"visual",header:"def invert(d):",chunks:[{prompt:"",indent:0},{prompt:"",indent:0},{prompt:"",indent:0}]});
    if(p.startsWith("/history/"))return setTimeout(()=>send({found:true,plan:graph,messages:[{role:"user",content:"I will count each value, keep those that appear once, then swap their keys and values.",at:new Date().toISOString()},{role:"assistant",content:"What should happen when two keys have the same value?",at:new Date().toISOString()}]}),1500);
    if(p==="/plan_graph")return send(graph);
    if(p==="/tutor_chat")return send({reply:"How does your plan handle an empty dictionary?",ready:false});
    if(p==="/design_review/plan"||p==="/design_review")return send({approved:true,reply:"Your approach is ready.",plan_graph:graph});
    if(p.startsWith("/session_steps/"))return send({chunks:[{prompt:"Collect the dictionary values.",indent:0},{prompt:"Identify values that appear once.",indent:0},{prompt:"Return the inverted dictionary.",indent:0}]});
    if(p==="/assignment_template")return send({template:"# Your Python assignment"});
    if(p.includes("/grades"))return send({students:[{student_id:"demo",name:"Alex Morgan",username:"demo@example.invalid",submitted:true,solved:2,shown:0,missed:4,total:6,percent:33}],problems:3,total_steps:6});
    if(p==="/playground/problems")return send({problems:[]});
    if(p==="/playground/params")return send({});
    return send({detail:"Unimplemented fixture"},404);
  }
  const target=path.resolve(frontend,"."+ (p==="/"?(role==="teacher"?"/teacher.html":"/dashboard.html"):p));
  if(!target.startsWith(frontend+path.sep)){res.writeHead(403).end();return;}
  fs.readFile(target,(err,data)=>{
    if(err){res.writeHead(404).end();return;}
    res.setHeader("Content-Type",{".html":"text/html",".js":"text/javascript",".css":"text/css",".svg":"image/svg+xml"}[path.extname(target)]||"text/plain");
    res.end(data);
  });
 }).listen(port,"0.0.0.0",()=>console.log("Synthetic "+role+" preview: http://localhost:"+port));
}
serve("student",8123);serve("teacher",8124);
