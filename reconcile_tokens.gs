// reconcile_tokens.gs
// Match part orders against Gate tokens.
// Requires two sheets:
//   - Gate_Responses (from proof_to_order_gate.gs)
//   - Orders (export from Strider): Date | Case_ID | SKU | ST_ID | Notes | ...
// Writes a "MissingToken" report to a sheet named "Recon".

function reconcileTokens(){
  const ss = SpreadsheetApp.getActive();
  const gate = ss.getSheetByName("Gate_Responses");
  const orders = ss.getSheetByName("Orders");
  const recon = ss.getSheetByName("Recon") || ss.insertSheet("Recon");

  const gh = gate.getRange(1,1,1,gate.getLastColumn()).getValues()[0];
  const oh = orders.getRange(1,1,1,orders.getLastColumn()).getValues()[0];

  const gateRows = gate.getRange(2,1,gate.getLastRow()-1,gate.getLastColumn()).getValues()
    .map(r => Object.fromEntries(gh.map((h,i)=>[h, r[i]])))
    .filter(r => (""+r.PassFail).toUpperCase() === "PASS");

  const orderRows = orders.getRange(2,1,orders.getLastRow()-1,orders.getLastColumn()).getValues()
    .map(r => Object.fromEntries(oh.map((h,i)=>[h, r[i]])));

  // Build lookup by Case_ID (or use Case_ID+SKU for stricter match)
  const okTokensByCase = {};
  gateRows.forEach(r => {
    const k = (""+r.Case_ID).trim();
    if (!okTokensByCase[k]) okTokensByCase[k] = [];
    okTokensByCase[k].push( (""+r.Token).trim() );
  });

  const missing = [];
  orderRows.forEach(o => {
    const caseId = (""+o.Case_ID).trim();
    const notes = (""+o.Notes).toUpperCase();
    const has = okTokensByCase[caseId] && okTokensByCase[caseId].some(t => t && notes.indexOf(t) >= 0);
    if (!has){
      missing.push({
        Date: o.Date,
        Case_ID: caseId,
        SKU: o.SKU,
        ST_ID: o.ST_ID,
        Notes: o.Notes,
        Status: "ORDER WITHOUT VALID GATE TOKEN"
      });
    }
  });

  // Write recon
  recon.clear();
  const hdr = ["Date","Case_ID","SKU","ST_ID","Notes","Status"];
  if (missing.length === 0){
    recon.getRange(1,1,1,hdr.length).setValues([hdr]);
    recon.getRange(2,1,1,1).setValue("All orders have valid tokens.");
    return;
  }
  const rows = missing.map(m => hdr.map(h => m[h] || ""));
  recon.getRange(1,1,1,hdr.length).setValues([hdr]);
  recon.getRange(2,1,rows.length,hdr.length).setValues(rows);
}
