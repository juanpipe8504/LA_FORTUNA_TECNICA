import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { makeAssistantMessage } from "../test/fixtures";
import { AssistantMessageView } from "./AssistantMessageView";

describe("AssistantMessageView", () => {
  it("renders markdown tables and citation badges linked to sources", () => {
    render(<AssistantMessageView message={makeAssistantMessage()} />);
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "8000" })).toBeInTheDocument();

    const badge = screen.getByRole("button", { name: /Ver Fuente 1: manual\.pdf, página 3/ });
    expect(badge).toHaveTextContent("1");
    expect(screen.getByRole("button", { name: /Ver Fuente 2: manual\.pdf, página 5/ })).toBeInTheDocument();
  });

  it("shows the visual context next to the answer with document/page badge", () => {
    render(<AssistantMessageView message={makeAssistantMessage()} />);
    const visual = screen.getByRole("complementary", { name: "Contexto visual" });
    const img = within(visual).getByRole("img", { name: "Figura 2: Diagrama de arquitectura" });
    expect(img).toHaveAttribute("src", "/api/v1/documents/doc-a/images/doc-a_p5_i0");
    expect(within(visual).getByText("manual.pdf")).toBeInTheDocument();
    expect(within(visual).getByText("Página 5")).toBeInTheDocument();
    // The figure links back to the source chunk it belongs to.
    expect(within(visual).getByRole("button", { name: "Ver fuente [2]" })).toBeInTheDocument();
  });

  it("highlights the matching source card when a citation is clicked", async () => {
    const user = userEvent.setup();
    render(<AssistantMessageView message={makeAssistantMessage()} />);
    const cards = screen.getAllByRole("listitem");
    expect(cards[1]).toHaveTextContent("Fuente: manual.pdf, Página 5");
    await user.click(screen.getByRole("button", { name: /Ver Fuente 2/ }));
    expect(cards[1]).toHaveClass("source-card--highlight");
    expect(cards[0]).not.toHaveClass("source-card--highlight");
  });

  it("opens a lightbox with caption and page, and closes it with Escape", async () => {
    const user = userEvent.setup();
    render(<AssistantMessageView message={makeAssistantMessage()} />);
    await user.click(screen.getByRole("button", { name: /Ampliar imagen: Figura 2/ }));
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("Página 5")).toBeInTheDocument();
    expect(within(dialog).getByText("Figura 2: Diagrama de arquitectura")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Cerrar (Esc)" })).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("degrades gracefully when an image fails to load", () => {
    render(<AssistantMessageView message={makeAssistantMessage()} />);
    fireEvent.error(screen.getByRole("img", { name: "Figura 2: Diagrama de arquitectura" }));
    expect(screen.getByRole("img", { name: "Imagen no disponible" })).toBeInTheDocument();
    // The page reference stays visible even without the image.
    expect(screen.getByText("Página 5")).toBeInTheDocument();
  });

  it("uses a distinct warning style and hides images when context is insufficient", () => {
    const { container } = render(
      <AssistantMessageView
        message={makeAssistantMessage({
          answer: "[[NO_CONTEXT]] No tengo información suficiente en los documentos.",
          has_sufficient_context: false,
          sources: [],
        })}
      />,
    );
    expect(screen.getByText("Información insuficiente")).toBeInTheDocument();
    expect(container.querySelector(".message--insufficient")).not.toBeNull();
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
    expect(screen.queryByText(/NO_CONTEXT/)).not.toBeInTheDocument();
  });
});
