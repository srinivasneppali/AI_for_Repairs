// proof_to_order_gate.gs
// V1 — 60-second gate for evidence + 2 checks using Google Form + Sheet.
// On form submit, this script validates answers, generates a Gate Token,
// and writes it back to the response row.
//
// Expected columns in "Gate_Responses" (edit to match your form headers):
// Timestamp | ST_ID | Case_ID | SKU | Fault_Code | Q1_PressureRelieved | Q2_TapeWraps | Q3_TorqueBand | Photo_URL | Video_URL | Token | PassFail | Notes

const SHEET_NAME = "Gate_Responses";
const PASS_STR = "PASS";
const FAIL_STR = "FAIL";

function onFormSubmit(e) {
  const sh = SpreadsheetApp.getActive().getSheetByName(SHEET_NAME);
  const hdr = sh.getRange(1,1,1,sh.getLastColumn()).getValues()[0];
  const row = e.range.getRow();
  const data = sh.getRange(row,1,1,sh.getLastColumn()).getValues()[0];
  const rec = Object.fromEntries(hdr.map((h,i)=>[h, data[i]]));

  // Basic validation rules (edit as needed)
  const q1_ok = (""+rec.Q1_PressureRelieved).toLowerCase().startsWith("y"); // Yes
  const q2_ok = (""+rec.Q2_TapeWraps).trim() === "6";                       // 6 wraps
  const q3_ok = (""+rec.Q3_TorqueBand).indexOf("1.2") >= 0;                 // 1.2–1.8 N·m band
  const photo_ok = (""+rec.Photo_URL).length > 5;

  const pass = q1_ok && q2_ok && q3_ok && photo_ok;
  const token = pass ? generateToken(rec) : "";

  // Write back Token and Pass/Fail
  const tokenCol = hdr.indexOf("Token")+1;
  const pfCol = hdr.indexOf("PassFail")+1;
  sh.getRange(row, tokenCol).setValue(token);
  sh.getRange(row, pfCol).setValue(pass ? PASS_STR : FAIL_STR);

  // Optional notification on FAIL
  // if (!pass) MailApp.sendEmail("trainer@example.com", "Gate FAIL "+rec.Case_ID, JSON.stringify(rec, null, 2));
}

function generateToken(rec){
  // Token pattern: FAULT-SKU-RAND
  const rand = Math.random().toString(36).slice(2,7).toUpperCase();
  const fault = (""+rec.Fault_Code).slice(0,3).toUpperCase();
  const sku = (""+rec.SKU).slice(-3).toUpperCase();
  return fault+"-"+sku+"-"+rand;
}
