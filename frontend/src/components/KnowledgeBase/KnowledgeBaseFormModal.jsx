import { useState } from "react";
import { Modal } from "../ui/Modal";
import { Button } from "../ui/Button";
import { Field, Input, Textarea } from "../ui/Field";
import { getErrorMessage } from "../../utils/helpers";

/** Shared create / rename dialog for knowledge bases. */
export const KnowledgeBaseFormModal = ({ mode = "create", initialValues, onSubmit, onClose }) => {
  const [name, setName] = useState(initialValues?.name || "");
  const [description, setDescription] = useState(initialValues?.description || "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const isEdit = mode === "edit";

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim() || saving) return;
    setSaving(true);
    setError(null);
    try {
      await onSubmit({ name: name.trim(), description: description.trim() || null });
      onClose();
    } catch (err) {
      setError(getErrorMessage(err));
      setSaving(false);
    }
  };

  return (
    <Modal
      title={isEdit ? "Edit knowledge base" : "New knowledge base"}
      description={isEdit ? undefined : "A knowledge base groups documents so chats can search only what you choose."}
      onClose={saving ? () => {} : onClose}
      dismissible={!saving}
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <Field label="Name">
          {(p) => (
            <Input
              {...p}
              data-autofocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={255}
              placeholder="e.g. Cloud computing notes"
              required
            />
          )}
        </Field>
        <Field label="Description" optional>
          {(p) => (
            <Textarea
              {...p}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              maxLength={2000}
              rows={3}
              placeholder="What kind of documents live here?"
            />
          )}
        </Field>
        {error && <p role="alert" className="rounded-lg border border-danger/30 bg-danger-soft px-3 py-2 text-sm text-danger">{error}</p>}
        <div className="flex justify-end gap-2 pt-1">
          <Button variant="secondary" onClick={onClose} disabled={saving}>Cancel</Button>
          <Button type="submit" disabled={!name.trim()} loading={saving}>
            {isEdit ? "Save changes" : "Create"}
          </Button>
        </div>
      </form>
    </Modal>
  );
};
