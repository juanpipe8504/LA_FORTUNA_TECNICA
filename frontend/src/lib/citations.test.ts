import { describe, expect, it } from "vitest";
import { buildCitationMap, cleanAnswer, expandCitationGroup, parseCitationNumbers } from "./citations";

describe("parseCitationNumbers", () => {
  it("extracts distinct numbers in order of appearance", () => {
    expect(parseCitationNumbers("A [2] B [1] C [2] D [3]")).toEqual([2, 1, 3]);
  });

  it("supports grouped and ranged citations", () => {
    expect(parseCitationNumbers("Ver [1, 3] y [4-6].")).toEqual([1, 3, 4, 5, 6]);
    expect(expandCitationGroup("2;5")).toEqual([2, 5]);
  });

  it("ignores markdown links, markers and code", () => {
    const md = "Un [enlace](http://x) y `arr[0]` y [[NO_CONTEXT]]\n```\nx[1]\n```\nfin [7]";
    expect(parseCitationNumbers(md)).toEqual([7]);
  });
});

describe("buildCitationMap", () => {
  it("maps [n] to sources[n-1] when all citations fit", () => {
    const { toSource, toCitation } = buildCitationMap("a [1] b [2]", 3);
    expect(toSource.get(1)).toBe(0);
    expect(toSource.get(2)).toBe(1);
    // The uncited third source gets the next free number.
    expect(toCitation.get(2)).toBe(3);
  });

  it("maps sorted context numbers positionally when backend kept only cited chunks", () => {
    const { toSource, toCitation } = buildCitationMap("x [4] y [1]", 2);
    expect(toSource.get(1)).toBe(0);
    expect(toSource.get(4)).toBe(1);
    expect(toCitation.get(1)).toBe(4);
  });

  it("leaves citations without a matching source unresolved", () => {
    const { toSource } = buildCitationMap("x [1] [5] [9]", 2);
    expect(toSource.get(1)).toBe(0);
    expect(toSource.has(5)).toBe(false);
  });

  it("numbers sources positionally when the answer has no citations", () => {
    const { toCitation } = buildCitationMap("sin citas", 2);
    expect([toCitation.get(0), toCitation.get(1)]).toEqual([1, 2]);
  });
});

it("cleanAnswer strips the insufficient-context marker", () => {
  expect(cleanAnswer("[[NO_CONTEXT]] No tengo información.")).toBe("No tengo información.");
});
