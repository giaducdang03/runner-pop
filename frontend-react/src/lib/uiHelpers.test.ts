import { describe, expect, it } from "vitest";
import { engineReady, pickDefaultEngine, prepareNote, shouldAutoRun, shouldPostPrepare } from "./uiHelpers";

describe("uiHelpers", () => {
  it("engineReady chỉ chặn idle/running", () => {
    expect(engineReady("done")).toBe(true);
    expect(engineReady("error")).toBe(true);
    expect(engineReady(undefined)).toBe(true);
    expect(engineReady("idle")).toBe(false);
    expect(engineReady("running")).toBe(false);
  });
  it("chỉ POST prepare khi idle", () => {
    expect(shouldPostPrepare("idle")).toBe(true);
    expect(shouldPostPrepare("error")).toBe(false);
  });
  it("engine mặc định", () => {
    expect(pickDefaultEngine({ default_engine: "cf" }, ["a", "cf"])).toBe("cf");
    expect(pickDefaultEngine({ default_engine: "x" }, ["a", "cf"])).toBe("a");
  });
  it("auto-run một lần cho mỗi chữ ký", () => {
    expect(shouldAutoRun(true, true, "s1", null, false)).toBe(true);
    expect(shouldAutoRun(true, true, "s1", "s1", false)).toBe(false);
    expect(shouldAutoRun(true, true, "s2", "s1", true)).toBe(false);
  });
  it("ghi chú fallback CPU", () => {
    expect(prepareNote("gpu", "CPU")).not.toBe("");
    expect(prepareNote("auto", "CPU")).toBe("");
  });
});
