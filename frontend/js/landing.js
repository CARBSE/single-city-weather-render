const startBtn = document.getElementById('startBtn');
const statusText = document.getElementById('statusText');
async function checkHealth(){
  try {
    const r = await fetch('/health', {cache:'no-store'});
    if (r.ok) { statusText.textContent='ready'; startBtn.textContent='Open app'; return true; }
  } catch(e){}
  statusText.textContent='starting';
  return false;
}
async function poll() {
  startBtn.disabled = true; startBtn.textContent = 'Starting…';
  while(true){
    const ok = await checkHealth();
    if(ok) { startBtn.disabled=false; break; }
    await new Promise(r=>setTimeout(r,3000));
  }
}
startBtn.addEventListener('click', async ()=>{
  if (startBtn.textContent === 'Open app') { window.location.href = '/'; return; }
  fetch('/', {cache:'no-store'}).catch(()=>{});
  poll();
});
window.addEventListener('load', ()=>checkHealth());
