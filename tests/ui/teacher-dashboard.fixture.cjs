/* Synthetic class data shared by DOM, browser and local preview checks. */
const assignments = [{id:'lab1',name:'LAB1 · Dictionaries',published:true},
  {id:'hw3',name:'HW3 · Stacks and calculators',published:true}];
const names = ['Alex Morgan','Jordan Lee','Sam Rivera','Taylor Chen','Jamie Patel',
  'Casey Brooks','Morgan Kim','Riley Davis'];
const problems = [
  ['invert','Invert a dictionary','lab1',16,8,5],
  ['postfix','Evaluate postfix expressions','hw3',13,5,4],
  ['push','Push to a stack','hw3',14,3,6],
  ['count','Count unique values','lab1',16,2,3],
  ['pop','Pop from a stack','hw3',10,1,4],
  ['peek','Peek at the top','hw3',9,0,4],
  ['empty','Check an empty stack','hw3',16,0,2],
  ['length','Count stack elements','hw3',12,0,0],
].map(([slug,title,assignment_id,attempted,needs_help,recovered])=>({
  slug,title,assignment_id,opened:attempted,attempted,needs_help,recovered,
  difficulty_percent:Math.round((needs_help+recovered)/attempted*100),
  steps:[{number:2,attempted,needs_help,recovered,prompt:'Return the value on top',prompt_varies:true},
         {number:1,attempted,needs_help:0,recovered:2,prompt:'Handle an empty stack',prompt_varies:false}],
  follow_up:names.slice(0,needs_help).map((name,i)=>{
    const reason=i%2?'Your solution runs but gives the wrong answer on at least one case.':'Your step runs, but the finished solution gives the wrong answer.';
    return {student_id:'student-'+i,name,steps:[2],reason,last_activity:'2026-09-16T12:00:00Z',
      details:[{number:2,prompt:'Return the value on top',reason,
                code:'if self.top is None:\n    return None\nreturn self.top',at:'2026-09-16T12:00:00Z'}]};})
}));
module.exports = (assignment_id=null) => {
  const selected=problems.filter(p=>!assignment_id||p.assignment_id===assignment_id);
  return structuredClone({generated_at:'2026-09-16T12:00:00Z',assignments,assignment_id,problems:selected,
    summary:{students:18,active:16,needs_help:Math.max(0,...selected.map(p=>p.needs_help)),
      problems:selected.length,attempted_problems:selected.length,indeterminate:1}});
};
